# Detect_IT 🕵️‍♂️📰 [whole notebook: https://colab.research.google.com/drive/17cxBmfExWanLScVxiTu8JL850SScvo-o?usp=drive_link]

**Detect_IT** is an intelligent browser companion and Machine Learning backend that actively scans news articles you read to detect and flag Fake News in real-time, utilizing fine-tuned NLP BERT classifications.

## Features
- **Highly Accurate Detection:** Under the hood, Detect_IT uses Google's pre-trained deep learning language model (BERT).
- **Real-Time Web Scanning:** Works autonomously as you surf platforms.
- **Offline Backend Inference:** Operates on a locally hosted REST server, ensuring total user data privacy.

---

## 🚀 Installation & Setup Guide

Since this project bridges Python Deep Learning and a Chrome extension, you need to execute two broad setup steps: running the backend server and loading the Chrome UI.

### Step 1: Python Backend Initialization

1. **Clone the Repository**
   ```bash
   git clone https://github.com/All-00k/Detect_it.git
   cd Detect_it
   ```

2. **Setup your Environment**
   It's highly recommended to use a virtual environment.
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate 
   # On Mac/Linux: source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get Data from the Drive link to train model**
   **Drive Link:** https://drive.google.com/drive/folders/1RTVRrzxAdqMfBUhCuE2dhNYiOCVJash0?usp=drive_link

5. **Prepare Model Weights (Optional if provided)**
   If model weights `detect_it_weights.pt` are not bundled via Git LFS, you must train the model locally by placing `a1_True.csv` and `a2_Fake.csv` inside the `data/` folder and running:
   ```bash
   python train.py
   ```
   *Note: Using a machine with a dedicated GPU (CUDA) is highly recommended for retraining.*

   
6. **Spin up the Server**
   Start the local Flask prediction server. This *must* be running continuously while the extension is active.
   ```bash
   python server.py
   ```
   You should see `API running at http://127.0.0.1:8765`.

### Step 2: Install Browser Extension

1. **Open Google Chrome** and travel to `chrome://extensions/` in your search bar.
2. Ensure **Developer Mode** toggle (top-right corner) is turned **ON**.
3. Click the **Load unpacked** button.
4. In the file navigator, select the `extension/` folder located precisely inside your cloned `detect_it` directory.
5. *Success!* A new icon widget will be nested alongside your Chrome Extensions tab.

---

## 🛠️ Usage Workflow

1. Keep the python server active in your terminal console.
2. Navigate casually to any online news publication or blog (e.g. CNN, Fox, Medium).
3. The Chrome extension will query the article's text, pass it to your local BERT `server.py`, evaluate the vocabulary structure, and ultimately prompt a floating metric revealing its computed probability of being **REAL** vs **FAKE**!

Enjoy scanning with Detect_IT!





