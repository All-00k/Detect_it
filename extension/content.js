/**
 * content.js — Content script injected into every page by Detect_IT
 *
 * Extracts readable article text when requested by the popup via background.js.
 * Strategy:
 *   1. Try <article> tag first (semantic HTML)
 *   2. Fall back to common blog/news content selectors
 *   3. Fall back to all <p> tags
 * Strips scripts, styles, ads, nav, footer automatically.
 */

(function () {
    "use strict";

    /**
     * Extract clean readable text from the current page.
     * Returns up to ~8000 characters (plenty for BERT 512-token window).
     */
    function extractArticleText() {
        // Selectors in priority order
        const selectors = [
            "article",
            '[role="main"]',
            ".post-content",
            ".entry-content",
            ".article-body",
            ".article-content",
            ".story-body",
            ".content-body",
            ".blog-post",
            "main",
            "#content",
            ".content",
        ];

        let container = null;
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) { container = el; break; }
        }

        // Fall back: collect all <p> tags
        if (!container) {
            const paragraphs = Array.from(document.querySelectorAll("p"))
                .map(p => p.innerText.trim())
                .filter(t => t.length > 40);
            return paragraphs.slice(0, 50).join(" ").slice(0, 8000);
        }

        // Clone and remove noise elements
        const clone = container.cloneNode(true);
        const noise = ["script", "style", "nav", "footer", "aside",
            "figure", "figcaption", "iframe", "noscript",
            ".ad", ".ads", ".advertisement", ".social-share",
            ".related-posts", ".sidebar", "#sidebar", "header"];
        noise.forEach(sel => {
            clone.querySelectorAll(sel).forEach(el => el.remove());
        });

        const text = clone.innerText || clone.textContent || "";
        return text.replace(/\s+/g, " ").trim().slice(0, 8000);
    }

    // Listen for messages from background.js / popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === "EXTRACT_TEXT") {
            try {
                const text = extractArticleText();
                const title = document.title || "";
                const url = window.location.href;
                sendResponse({ text, title, url, charCount: text.length });
            } catch (err) {
                sendResponse({ error: err.message });
            }
        }
        return true;
    });
})();
