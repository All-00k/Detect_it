# Detect_IT System Architecture & Workflow

## Overview
Detect_IT is an end-to-end Machine Learning based application built to classify news articles as **Real** or **Fake**. It leverages a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model fine-tuned for text classification, a Flask backend server to handle inferences, and a Chrome extension to inject real-time safety scores while users browe the web.

## System Architecture

The architecture follows a standard offline-training to online-inference model deployed in a Client-Server topology.

1. **Training Engine (Offline Phase)**: Reads large datasets of known Real and Fake news, tokenizes the text via Hugging Face `transformers`, and fine-tunes a PyTorch BERT classification head using GPU-accelerated computing.
2. **Inference Backend (Online Phase)**: A Flask server that loads the serialized model weights into memory and exposes a RESTful POST endpoint (`/predict`). 
3. **Browser Extention (Client Phase)**: A Chrome extension running via JavaScript that scrapes article text from the user's active tab, passes it to the Flask backend, and renders a floating warning/trust score badge.

## File Hierarchy and Needs

- **`model.py`**: Defines the custom PyTorch NLP architecture (`BERT_Arch`). It essentially downloads the base BERT structure and appends neural classification layers on top of it.
- **`train.py`**: The main execution script for training. It parses CSV data, runs the Hugging Face tokenizer, iteratively calculates loss (NLLLoss/AdamW optimizer), and exports the optimal state dictionaries (weights).
- **`server.py`**: A continuously running Flask web service. It imports the model, waits for requests over `localhost:8765`, transforms string requests to tensors, evaluates them, and returns JSON probabilities.
- **`data/`**: Directory meant to house raw text data files like `a1_True.csv` and `a2_Fake.csv` that supply truth-labels for supervised learning.
- **`models/`**: House the heavy (`.pt`) model weight matrices generated after successful training computations.
- **`extension/`**: Contains the Chrome Extension user-interface scripts (Manifest, HTML popups, icons, context-scripts).
- **`notebooks/`**: Dedicated sandbox area containing `.ipynb` files used strictly for exploratory data analysis (EDA) and rapid algorithmic prototyping.
- **`scripts/`**: Houses utility macros over the project lifespan (e.g. `generate_icons.py` to auto-brush app UI variants).
- **`requirements.txt`**: Standard Python environment dependency ledger.

## Step-by-Step Workflow (Our Development History)

*(Below describes the hypothetical timeline tracing from project inception to complete training and web deployment)*

1. **Data Acquisition & Setup**: We gathered 100,000+ distinct news articles and classified them neatly into `a1_True` and `a2_Fake` datasets securely inside the `data/` branch.
2. **Experimentation**: Utilizing `b1_BERT_Walkthrough.ipynb` and `x1_FakeNewDetection.ipynb` in the `notebooks/` directory, we proved the feasibility of contextual NLP against our text corpus.
3. **Architecture Mapping**: We transferred code from messy sandboxes into a rigid Class-based architecture inside `model.py` locking the BERT base and setting up an active head classifier.
4. **Offline Training Completion**: 
   - We ran `python train.py`. 
   - The tokenization successfully passed into dataloaders. Over the course of designated epochs, the model optimized its parameters. 
   - Once validated, the terminal logged success and deployed the robust inference file `detect_it_weights.pt` precisely into the `models/` directory.
5. **API Deployment**: We established `server.py` to dynamically serve these newly tuned tensor weights entirely over an HTTP socket.
6. **Integration**: The user-facing Chrome add-on was polished in the `extension/` branch to query this local port autonomously on web-navigation.
7. **Refactoring & Cleanup**: Finally, we centralized orphaned files, discarded unused temporary environments, and enforced a scalable directory hierarchy to transition the project into a professional, production-ready product state.
