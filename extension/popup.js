const BACKEND_URL = "http://127.0.0.1:8000"; // change to your hosted URL for production

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultDiv = document.getElementById("result");

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  resultDiv.textContent = "Checking…";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${BACKEND_URL}/check-image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Server responded with ${response.status}`);

    const data = await response.json();
    const pct = Math.round(data.confidence * 100);

    resultDiv.innerHTML = `
      <strong>${data.verdict}</strong> — ${pct}% confidence<br/>
      <span style="color:#555">${data.explanation}</span>
    `;
  } catch (err) {
    resultDiv.textContent = `Error: ${err.message}. Is the backend running?`;
  }
});
