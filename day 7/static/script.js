const form = document.getElementById('prediction-form');
const predictionLabel = document.getElementById('prediction-label');
const predictionNote = document.getElementById('prediction-note');
const probabilityList = document.getElementById('probability-list');

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = '';

  const entries = Object.entries(probabilities || {});

  if (!entries.length) {
    probabilityList.innerHTML = '<p class="empty-state">No probabilities returned.</p>';
    return;
  }

  entries.forEach(([label, probability]) => {
    const percent = Math.round(probability * 1000) / 10;
    const item = document.createElement('div');
    item.className = 'probability-item';
    item.innerHTML = `
      <div class="probability-row">
        <strong>${label}</strong>
        <span>${percent}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${percent}%"></div>
      </div>
    `;
    probabilityList.appendChild(item);
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  predictionLabel.textContent = 'Predicting...';
  predictionNote.textContent = 'The saved model is evaluating the submitted student profile.';
  probabilityList.innerHTML = '<p class="empty-state">Loading probabilities...</p>';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error('Prediction request failed.');
    }

    const data = await response.json();
    predictionLabel.textContent = data.prediction;
    predictionNote.textContent = `Highest model confidence: ${Math.round((data.confidence || 0) * 1000) / 10}%`;
    renderProbabilities(data.probabilities);
  } catch (error) {
    predictionLabel.textContent = 'Prediction unavailable';
    predictionNote.textContent = error.message;
    probabilityList.innerHTML = '<p class="empty-state">Check the server and try again.</p>';
  }
});

form.addEventListener('reset', () => {
  predictionLabel.textContent = 'Waiting for input';
  predictionNote.textContent = 'Submit the form to see the predicted learning outcome.';
  probabilityList.innerHTML = '';
});