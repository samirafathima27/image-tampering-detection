// Background service worker
// Sets up the right-click context menu on images, sends the clicked
// image URL to the backend API, and forwards the result to the content
// script to render the overlay card.

const BACKEND_URL = "http://127.0.0.1:8000"; // change to your hosted URL for production

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-image-authenticity",
    title: "Check Image Authenticity",
    contexts: ["image"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-image-authenticity") return;

  const imageUrl = info.srcUrl;

  // Tell the content script to show a "checking..." loading card immediately
  chrome.tabs.sendMessage(tab.id, {
    type: "SHOW_LOADING",
    imageUrl,
  });

  try {
    const response = await fetch(`${BACKEND_URL}/check-image-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: imageUrl }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const result = await response.json();

    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_RESULT",
      imageUrl,
      result, // { verdict, confidence, explanation, heatmap_base64 }
    });
  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_ERROR",
      imageUrl,
      error: err.message,
    });
  }
});
