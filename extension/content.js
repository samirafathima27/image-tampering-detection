// Content script
// Listens for messages from background.js and renders a floating result
// card directly on the page, near the checked image.

let currentCard = null;

function removeCard() {
  if (currentCard) {
    currentCard.remove();
    currentCard = null;
  }
}

function createCard(html) {
  removeCard();
  const card = document.createElement("div");
  card.className = "tamper-check-card";
  card.innerHTML = html;
  document.body.appendChild(card);
  currentCard = card;

  const closeBtn = card.querySelector(".tamper-check-close");
  if (closeBtn) closeBtn.addEventListener("click", removeCard);

  return card;
}

function verdictClass(verdict) {
  if (verdict === "Real") return "verdict-real";
  if (verdict === "Photoshop-tampered") return "verdict-photoshop";
  if (verdict === "AI-tampered") return "verdict-ai";
  return "";
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "SHOW_LOADING") {
    createCard(`
      <div class="tamper-check-header">
        <span>Checking image authenticity…</span>
        <button class="tamper-check-close">&times;</button>
      </div>
      <div class="tamper-check-body">
        <div class="tamper-check-spinner"></div>
      </div>
    `);
  }

  if (message.type === "SHOW_RESULT") {
    const { verdict, confidence, explanation, heatmap_base64 } = message.result;
    const pct = Math.round(confidence * 100);

    createCard(`
      <div class="tamper-check-header">
        <span>Image Authenticity Result</span>
        <button class="tamper-check-close">&times;</button>
      </div>
      <div class="tamper-check-body">
        <div class="tamper-check-verdict ${verdictClass(verdict)}">
          ${verdict} — ${pct}% confidence
        </div>
        <img class="tamper-check-heatmap" src="data:image/png;base64,${heatmap_base64}" alt="tampered region heatmap" />
        <p class="tamper-check-explanation">${explanation}</p>
      </div>
    `);
  }

  if (message.type === "SHOW_ERROR") {
    createCard(`
      <div class="tamper-check-header">
        <span>Error</span>
        <button class="tamper-check-close">&times;</button>
      </div>
      <div class="tamper-check-body">
        <p class="tamper-check-explanation">Could not check this image: ${message.error}</p>
        <p class="tamper-check-hint">Is the backend server running?</p>
      </div>
    `);
  }
});
