"""
server.py — Flask API server for Detect_IT Chrome Extension

Loads the fine-tuned BERT model and exposes:
  POST /predict   { "text": "article body..." }
  → { "label": "FAKE", "confidence": 87.3, "fake_prob": 87.3, "real_prob": 12.7 }

  GET  /health    → { "status": "ok", "model": "loaded" }

Usage:
    cd D:\\ForMe\\Detect_IT
    python server.py

Then load the Chrome extension and browse any news blog.
The server must be running for the extension to work.
"""

import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.dirname(__file__))
from model import BERT_Arch

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "detect_it_weights.pt")
FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "models", "c1_fakenews_weights.pt")
MAX_LEN      = 512
PORT         = 8765
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model at startup ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Detect_IT API Server")
print(f"{'='*60}")
print(f"  Device : {device}")
print(f"  Port   : {PORT}")
print(f"  Loading model …")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BERT_Arch()

if os.path.exists(WEIGHTS_PATH):
    print(f"  ✅ Loading Detect_IT weights: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
elif os.path.exists(FALLBACK_PATH):
    print(f"  ⚠️  detect_it_weights.pt not found.")
    print(f"  ✅ Loading fallback weights: {FALLBACK_PATH}")
    print(f"  (Run train.py to generate new weights)")
    model.load_state_dict(torch.load(FALLBACK_PATH, map_location=device), strict=False)
else:
    print("  ⚠️  No weights found! Model will produce random predictions.")
    print(f"  Run train.py first to generate weights.")

model.to(device)
model.eval()
print(f"  Model ready.\n{'='*60}\n")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])   # Allow Chrome extension origins


def predict_text(text: str) -> dict:
    """Run inference on a text string. Returns label + probabilities."""
    text = text.strip()
    if not text:
        return {"error": "Empty text"}

    encoding = tokenizer(
        [text],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    seq  = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(seq, mask)
        probs  = torch.exp(logits).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    fake_prob  = float(probs[1]) * 100
    real_prob  = float(probs[0]) * 100
    label      = "FAKE" if pred_idx == 1 else "REAL"
    confidence = fake_prob if pred_idx == 1 else real_prob

    return {
        "label":      label,
        "confidence": round(confidence, 2),
        "fake_prob":  round(fake_prob, 2),
        "real_prob":  round(real_prob, 2),
    }


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON body: {\"text\": \"...\"}"}), 400

    text   = data["text"]
    result = predict_text(text)

    # Log to console
    preview = text[:80].replace("\n", " ")
    print(f"[PREDICT] {result['label']} ({result['confidence']:.1f}%)  '{preview}…'")
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "loaded", "device": str(device)})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":    "Detect_IT API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Classify text as REAL or FAKE",
            "GET  /health":  "Health check"
        }
    })


if __name__ == "__main__":
    print(f"  API running at http://127.0.0.1:{PORT}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
