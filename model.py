"""
model.py — Shared BERT_Arch definition for Detect_IT
Used by both train.py (training) and server.py (inference).

Architecture:
  bert-base-uncased (frozen) →
  Dropout(0.1) → Linear(768→512) → ReLU → Dropout(0.1) → Linear(512→2) → LogSoftmax

Labels: 0 = Real, 1 = Fake
"""

import torch.nn as nn
from transformers import AutoModel


class BERT_Arch(nn.Module):
    def __init__(self):
        super(BERT_Arch, self).__init__()
        self.bert    = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def freeze_bert(self):
        """Freeze all BERT parameters — only train the classification head."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, sent_id, mask):
        # pooler_output: [CLS] token embedding → shape (batch, 768)
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
