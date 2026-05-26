/**
 * background.js — Service worker for Detect_IT (Manifest V3 requirement)
 * Acts as a message relay between popup and content scripts.
 */

chrome.runtime.onInstalled.addListener(() => {
    console.log("[Detect_IT] Extension installed and ready.");
});

// Relay messages from popup → content script → popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "EXTRACT_TEXT") {
        // Ask the active tab's content script to extract the page text
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (!tabs[0]) {
                sendResponse({ error: "No active tab found" });
                return;
            }
            chrome.tabs.sendMessage(tabs[0].id, { type: "EXTRACT_TEXT" }, (response) => {
                if (chrome.runtime.lastError) {
                    sendResponse({ error: chrome.runtime.lastError.message });
                } else {
                    sendResponse(response);
                }
            });
        });
        return true; // Keep message channel open for async response
    }
});
