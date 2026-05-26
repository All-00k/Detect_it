"""
Fake News Detection App
Uses a fine-tuned BERT model to classify news headlines as REAL or FAKE.
Based on b2_FakeNewDetection.ipynb by Skillcate AI.

Architecture:
  bert-base-uncased (frozen) → Dropout(0.1) → ReLU → Linear(768→512) → Dropout → Linear(512→2) → LogSoftmax
  Labels: 0 = Real, 1 = Fake

Run:
  pip install gradio transformers torch
  python fake_news_app.py
"""

import os
import sys
import numpy as np

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:
    import torch
    import torch.nn as nn
except ImportError:
    missing.append("torch")

try:
    from transformers import AutoModel, BertTokenizerFast
except ImportError:
    missing.append("transformers")

try:
    import gradio as gr
except ImportError:
    missing.append("gradio")

if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print(f"Install with:  pip install {' '.join(missing)}")
    sys.exit(1)

# ── Model Definition (must match training architecture exactly) ────────────────
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(0.1)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# ── Load model ────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_fakenews_weights.pt")
MAX_LEN      = 15
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {device}")
print("[INFO] Loading bert-base-uncased …")

bert      = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model     = BERT_Arch(bert)

if os.path.exists(WEIGHTS_PATH):
    print(f"[INFO] Loading fine-tuned weights from {WEIGHTS_PATH}")
    state = torch.load(WEIGHTS_PATH, map_location=device)
    result = model.load_state_dict(state, strict=False)
    print(f"[INFO] ✅ Weights loaded. Missing: {result.missing_keys}  Unexpected: {result.unexpected_keys}")
else:
    print(f"[WARNING] Weight file not found at {WEIGHTS_PATH}.")
    print("[WARNING] Running with un-fine-tuned classification head (predictions will be random).")

model.to(device)
model.eval()

# ── Sample headlines ───────────────────────────────────────────────────────────
SAMPLES = [
    ("Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",   "FAKE"),
    ("WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",              "FAKE"),
    ("U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",          "REAL"),
    ("Trump administration issues new rules on U.S. visa waivers",                       "REAL"),
    ("BREAKING: Secret Recording Proves Obama Wiretapped Trump Campaign ENTIRE TIME",     "FAKE"),
    ("Senate passes $1.1 trillion spending bill to fund government through September",    "REAL"),
    ("Kremlin says no firm date set yet for proposed congressional visit",                "REAL"),
    ("DISGUSTING! Michelle Obama Caught On Camera Saying THIS About America",             "FAKE"),
]

# ── Inference function ─────────────────────────────────────────────────────────
def predict(headline: str):
    """Tokenize a headline and return (label, confidence, fake_prob, real_prob)."""
    headline = headline.strip()
    if not headline:
        return None, None, None, None

    # Modern transformers API: call tokenizer directly
    encoding = tokenizer(
        [headline],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    seq  = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(seq, mask)                   # log-softmax outputs
        probs  = torch.exp(logits).cpu().numpy()[0] # convert log-prob → prob

    pred_idx   = int(np.argmax(probs))
    fake_prob  = float(probs[1]) * 100
    real_prob  = float(probs[0]) * 100
    label      = "FAKE" if pred_idx == 1 else "REAL"
    confidence = fake_prob if pred_idx == 1 else real_prob
    return label, confidence, fake_prob, real_prob


def run_prediction(headline: str):
    if not headline or not headline.strip():
        return ("<div style='text-align:center;color:#555;font-size:1.1em;padding:3em'>"
                "Enter a news headline above and click <b>Analyze</b></div>")

    label, confidence, fake_prob, real_prob = predict(headline)
    if label is None:
        return "<div style='color:red;padding:1em'>Error during prediction.</div>"

    if label == "FAKE":
        badge_color  = "#ff4757"
        badge_bg     = "rgba(255,71,87,0.12)"
        icon         = "🚨"
        verdict_text = "This headline appears to be <b>FAKE NEWS</b>"
    else:
        badge_color  = "#2ed573"
        badge_bg     = "rgba(46,213,115,0.12)"
        icon         = "✅"
        verdict_text = "This headline appears to be <b>REAL NEWS</b>"

    result_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 28px 32px;
        font-family: 'Segoe UI', Inter, sans-serif;
        color: #e0e0e0;
        border: 1px solid {badge_color}44;
        box-shadow: 0 0 30px {badge_color}22;
        margin-top: 8px;
    ">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
            <span style="font-size:2.8em; line-height:1">{icon}</span>
            <div>
                <div style="
                    display:inline-block;
                    background:{badge_bg};
                    color:{badge_color};
                    border:2px solid {badge_color};
                    border-radius:8px;
                    padding:4px 16px;
                    font-size:1.5em;
                    font-weight:800;
                    letter-spacing:2px;
                ">{label}</div>
                <div style="margin-top:8px; font-size:1em; color:#aaa">{verdict_text}</div>
            </div>
        </div>

        <div style="margin-bottom:12px">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#2ed573; font-weight:600">✅ REAL ({real_prob:.1f}%)</span>
                <span style="color:#ff4757; font-weight:600">🚨 FAKE ({fake_prob:.1f}%)</span>
            </div>
            <div style="
                background: #2a2a4a;
                border-radius: 999px;
                height: 14px;
                overflow: hidden;
            ">
                <div style="display:flex; height:100%;">
                    <div style="
                        width: {real_prob:.1f}%;
                        background: linear-gradient(90deg, #2ed573, #7bed9f);
                    "></div>
                    <div style="
                        width: {fake_prob:.1f}%;
                        background: linear-gradient(90deg, #ff6b81, #ff4757);
                    "></div>
                </div>
            </div>
        </div>

        <div style="
            margin-top:16px;
            padding:12px 16px;
            background:rgba(255,255,255,0.04);
            border-radius:10px;
            font-size:0.9em;
            color:#888;
        ">
            <b style="color:{badge_color}">Confidence:</b> {confidence:.1f}% &nbsp;|&nbsp;
            <b style="color:#aaa">Model:</b> BERT-base-uncased (fine-tuned) &nbsp;|&nbsp;
            <b style="color:#aaa">Input tokens:</b> max {MAX_LEN}
        </div>
    </div>
    """
    return result_html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
body { background: #0d0d1a !important; }
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto;
    font-family: 'Segoe UI', Inter, 'Helvetica Neue', sans-serif !important;
    background: #0d0d1a !important;
}
#headline-input textarea {
    background: #1a1a2e !important;
    color: #e0e0e0 !important;
    border: 1.5px solid #4a4a8a !important;
    border-radius: 12px !important;
    font-size: 1.05em !important;
    padding: 14px !important;
    transition: border-color 0.2s;
}
#headline-input textarea:focus {
    border-color: #7c6bff !important;
    box-shadow: 0 0 0 3px rgba(124,107,255,0.15) !important;
}
#analyze-btn {
    background: linear-gradient(135deg, #7c6bff, #5a4fcf) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(124,107,255,0.4) !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(124,107,255,0.55) !important;
}
.sample-btn button {
    background: #1e1e3a !important;
    color: #b0b0e0 !important;
    border: 1px solid #3a3a6a !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 50px !important;
}
.sample-btn button:hover {
    background: #2a2a50 !important;
    border-color: #6060aa !important;
    color: #e0e0ff !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="🔍 Fake News Detector", theme=gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
).set(
    body_background_fill="#0d0d1a",
    block_background_fill="#13132a",
    block_border_color="#2a2a4a",
    input_background_fill="#1a1a2e",
    button_primary_background_fill="#7c6bff",
)) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        text-align: center;
        padding: 40px 20px 24px;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 0 0 24px 24px;
        margin-bottom: 28px;
    ">
        <div style="font-size: 2.8em; margin-bottom: 10px;">🔍</div>
        <h1 style="
            font-size: 2.2em;
            font-weight: 900;
            background: linear-gradient(135deg, #a78bfa, #7c6bff, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 10px;
        ">Fake News Detector</h1>
        <p style="color: #8888aa; font-size: 1.05em; margin: 0;">
            Powered by <b style="color:#a78bfa">BERT-base-uncased</b> fine-tuned on 44,898 news articles
        </p>
        <div style="
            display: inline-flex;
            gap: 16px;
            margin-top: 16px;
            flex-wrap: wrap;
            justify-content: center;
        ">
            <span style="background:#1e1e3a; color:#7c6bff; border:1px solid #3a3a6a; border-radius:20px; padding:4px 14px; font-size:0.85em">88% Test Accuracy</span>
            <span style="background:#1e1e3a; color:#60a5fa; border:1px solid #3a3a6a; border-radius:20px; padding:4px 14px; font-size:0.85em">44,898 Training Samples</span>
            <span style="background:#1e1e3a; color:#2ed573; border:1px solid #3a3a6a; border-radius:20px; padding:4px 14px; font-size:0.85em">Binary Classification</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML('<div style="color:#9090c0; font-size:0.82em; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px">📰 News Headline</div>')
            headline_input = gr.Textbox(
                placeholder="Paste or type a news headline here…",
                lines=3,
                show_label=False,
                elem_id="headline-input",
            )
            analyze_btn = gr.Button("🔍 Analyze Headline", variant="primary", elem_id="analyze-btn")
            result_html = gr.HTML(
                value="<div style='text-align:center;color:#555;font-size:1.1em;padding:3em'>"
                      "Enter a news headline above and click <b>Analyze</b></div>",
            )

        with gr.Column(scale=2):
            gr.HTML('<div style="color:#9090c0; font-size:0.82em; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px">⚡ Sample Headlines</div>')
            gr.HTML('<div style="color:#666; font-size:0.82em; margin-bottom:10px">Click any headline to auto-fill the input</div>')

            for text, expected in SAMPLES:
                badge = "🚨 FAKE" if expected == "FAKE" else "✅ REAL"
                btn = gr.Button(
                    f"{badge}  {text[:70]}{'…' if len(text) > 70 else ''}",
                    size="sm",
                    elem_classes=["sample-btn"],
                )
                btn.click(fn=lambda t=text: t, inputs=[], outputs=[headline_input])

    # ── Model info accordion ──────────────────────────────────────────────────
    with gr.Accordion("📖 About the Model", open=False):
        gr.HTML("""
        <div style="
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px 24px;
            color: #b0b0cc;
            font-size: 0.95em;
            line-height: 1.7;
        ">
            <h3 style="color:#a78bfa; margin-top:0">Architecture</h3>
            <ul>
                <li><b>Base:</b> bert-base-uncased (BERT layers frozen during training)</li>
                <li><b>Head:</b> Dropout(0.1) → Linear(768→512) → ReLU → Dropout → Linear(512→2) → LogSoftmax</li>
                <li><b>Input:</b> News headline tokenized to max 15 tokens</li>
                <li><b>Output:</b> 0 = Real, 1 = Fake</li>
            </ul>
            <h3 style="color:#60a5fa">Training</h3>
            <ul>
                <li><b>Dataset:</b> 44,898 articles — 52.3% Fake / 47.7% Real</li>
                <li><b>Split:</b> 70% train / 15% val / 15% test</li>
                <li><b>Optimizer:</b> AdamW lr=1e-5 | <b>Epochs:</b> 2</li>
                <li><b>Test accuracy:</b> 88%  (F1=0.88, precision=0.88, recall=0.88)</li>
            </ul>
            <h3 style="color:#ffa502">⚠️ Limitations</h3>
            <p>Model is trained only on news titles (max 15 tokens). Confidence scores are
               softmax probabilities — not calibrated. Best for political news headlines
               similar to the training domain.</p>
        </div>
        """)

    # ── Wire events ────────────────────────────────────────────────────────────
    analyze_btn.click(fn=run_prediction, inputs=[headline_input], outputs=[result_html])
    headline_input.submit(fn=run_prediction, inputs=[headline_input], outputs=[result_html])

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🔍 Fake News Detector - BERT Fine-tuned")
    print("="*60)
    print(f"  Device  : {device}")
    print(f"  Weights : {WEIGHTS_PATH}")
    print(f"  Max len : {MAX_LEN} tokens")
    print("="*60 + "\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
