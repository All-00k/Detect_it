import sys
import os
import torch
import pandas as pd
import numpy as np

# Add project root to sys.path for importing model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from model import BERT_Arch

# Constants
WEIGHTS_PATH = os.path.join("models", "c1_fakenews_weights.pt")
DATA_TRUE = os.path.join("data", "a1_True.csv")
DATA_FAKE = os.path.join("data", "a2_Fake.csv")
MAX_LENGHT = 15 # Based on notebooks

def evaluate():
    print("Loading model and weights...")
    device = torch.device("cpu")
    model = BERT_Arch()
    # Using c1_fakenews_weights.pt as the primary model weights
    # Use strict=False to bypass position_ids mismatch issues often found in BERT versions
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
    model.to(device)
    model.eval()

    print("Loading and preparing data...")
    try:
        true_df = pd.read_csv(DATA_TRUE)
        fake_df = pd.read_csv(DATA_FAKE)
    except FileNotFoundError:
        print("Error: CSV files not found in data/ directory.")
        return

    true_df['Target'] = 'True'
    fake_df['Target'] = 'Fake'
    
    true_df['label'] = 0
    fake_df['label'] = 1
    
    # Combine and sample with same random state as notebooks
    data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=2018).reset_index(drop=True)
    
    # Split replicates the notebook's 70/15/15 split
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        data['title'], data['label'], 
        random_state=2018, 
        test_size=0.3, 
        stratify=data['Target']
    )
    
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, 
        random_state=2018, 
        test_size=0.5, 
        stratify=temp_labels
    )
    
    # Use a subset for speed in restricted environment
    subset_size = 500
    if len(test_text) > subset_size:
        test_text_sub = test_text.iloc[:subset_size]
        test_labels_sub = test_labels.iloc[:subset_size]
    else:
        test_text_sub = test_text
        test_labels_sub = test_labels
    
    print(f"Tokenizing {len(test_text_sub)} test samples...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens_test = tokenizer(
        test_text_sub.tolist(),
        max_length = MAX_LENGHT,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )
    
    test_seq = tokens_test['input_ids']
    test_mask = tokens_test['attention_mask']
    
    print("Running inference...")
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        
    preds = np.argmax(preds, axis=1)
    
    print("\nEvaluation Results (on test subset):")
    report = classification_report(test_labels_sub, preds, target_names=['Real', 'Fake'], digits=4)
    print(report)

if __name__ == "__main__":
    evaluate()
