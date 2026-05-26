/**
 * popup.js — Logic for Detect_IT Chrome extension popup
 *
 * Flow:
 *   1. Show current page title/URL
 *   2. On "Analyze" click → ask content.js to extract text
 *   3. POST text to Flask API at localhost:8765/predict
 *   4. Render REAL/FAKE result card with confidence bar
 */

const API_URL = "http://127.0.0.1:8765/predict";

const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const resultCard = document.getElementById("result-card");
const resultIcon = document.getElementById("result-icon");
const resultBadge = document.getElementById("result-badge");
const resultVerdict = document.getElementById("result-verdict");
const barReal = document.getElementById("bar-real");
const barFake = document.getElementById("bar-fake");
const realLbl = document.getElementById("real-lbl");
const fakeLbl = document.getElementById("fake-lbl");
const excerptEl = document.getElementById("excerpt");
const pageTitleEl = document.getElementById("page-title");
const pageUrlEl = document.getElementById("page-url");
const offlineCard = document.getElementById("offline-card");

// ── On popup open: show current tab info ─────────────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
        pageTitleEl.textContent = tabs[0].title || "Untitled";
        pageUrlEl.textContent = tabs[0].url || "";
    }
});

// ── Analyze button click ──────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
    setLoading(true);
    offlineCard.style.display = "none";
    resultEl.style.display = "none";

    // Step 1: Get text from content script via background
    let extractedData;
    try {
        extractedData = await sendMessageToBackground({ type: "EXTRACT_TEXT" });
    } catch (err) {
        setStatus("❌ Could not read page content. Try refreshing.");
        setLoading(false);
        return;
    }

    if (!extractedData || extractedData.error) {
        setStatus("❌ " + (extractedData?.error || "Could not extract text from this page."));
        setLoading(false);
        return;
    }

    const { text, title, charCount } = extractedData;
    if (!text || text.trim().length < 50) {
        setStatus("⚠️ Not enough readable text found on this page.");
        setLoading(false);
        return;
    }

    setStatus(`<span class="spinner"></span> Analyzing ${charCount} characters…`);

    // Step 2: POST to local Flask API
    let result;
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: (title + " " + text).slice(0, 6000) }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        result = await response.json();
    } catch (err) {
        offlineCard.style.display = "block";
        setStatus("");
        setLoading(false);
        return;
    }

    // Step 3: Render result
    renderResult(result, text);
    setLoading(false);
    setStatus("");
});

// ── Render result card ────────────────────────────────────────────────────────
function renderResult(result, rawText) {
    const isFake = result.label === "FAKE";

    resultCard.className = "result-card " + (isFake ? "fake" : "real");
    resultIcon.textContent = isFake ? "🚨" : "✅";

    resultBadge.textContent = result.label;
    resultBadge.className = "result-badge " + (isFake ? "fake" : "real");

    resultVerdict.textContent = isFake
        ? `This article appears to be FAKE NEWS (${result.confidence.toFixed(1)}% confidence)`
        : `This article appears to be REAL NEWS (${result.confidence.toFixed(1)}% confidence)`;

    realLbl.textContent = `✅ REAL (${result.real_prob.toFixed(1)}%)`;
    fakeLbl.textContent = `🚨 FAKE (${result.fake_prob.toFixed(1)}%)`;

    // Animate bar widths
    setTimeout(() => {
        barReal.style.width = result.real_prob.toFixed(1) + "%";
        barFake.style.width = result.fake_prob.toFixed(1) + "%";
    }, 50);

    // Show excerpt of analyzed text
    const preview = rawText.replace(/\s+/g, " ").trim().slice(0, 160);
    excerptEl.innerHTML = `<b>Analyzed text:</b> "${preview}…"`;

    resultEl.style.display = "block";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.textContent = loading ? "Analyzing…" : "🔍 Analyze This Page";
}

function setStatus(html) {
    statusEl.innerHTML = html;
}

function sendMessageToBackground(message) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(message, (response) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
            } else {
                resolve(response);
            }
        });
    });
}
