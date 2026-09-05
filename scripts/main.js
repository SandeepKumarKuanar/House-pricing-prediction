// app.js

const API_BASE_URL = 'https://house-pricing-prediction-fv7l.onrender.com';
const POLL_INTERVAL_MS = 2000; // Poll every 2 seconds
const MAX_POLL_ATTEMPTS = 45; // Give up after ~90 seconds
const FETCH_TIMEOUT_MS = 30000; // 30s timeout for the initial POST during cold start

const form = document.getElementById('prediction-form');
const resultDiv = document.getElementById('result');

function buildFormData() {
    return {
        sqft_living: parseFloat(document.getElementById('sqft_living').value),
        bedrooms: parseInt(document.getElementById('bedrooms').value),
        bathrooms: parseFloat(document.getElementById('bathrooms').value),
        // These are just example values for the features you don't have inputs for yet
        sqft_lot: 7912,
        floors: 1.5,
        waterfront: 0,
        view: 0,
        condition: 3,
        grade: 7,
        sqft_above: 1340,
        sqft_basement: 0,
        yr_built: 1955,
        yr_renovated: 0,
        zipcode: 98125,
        lat: 47.7210,
        long: -122.319,
        sqft_living15: 1690,
        sqft_lot15: 7639
    };
}

async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

async function pollForResult(taskId) {
    const statusUrl = `${API_BASE_URL}/api/status/${taskId}/`;
    for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));

        let response;
        try {
            response = await fetchWithTimeout(statusUrl, {}, FETCH_TIMEOUT_MS);
        } catch (error) {
            // Transient network error during polling; retry on the next tick
            console.error('Status poll failed:', error);
            continue;
        }

        if (response.status === 200) {
            return await response.json();
        }

        if (response.status === 404 || response.status === 500) {
            throw new Error(`Server error while checking status (${response.status})`);
        }

        // status 202 = still loading, keep polling
    }
    throw new Error('Prediction timed out. The server is taking too long. Try again.');
}

form.addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent the form from reloading the page
    resultDiv.textContent = 'Loading prediction...';

    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/api/predict/`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(buildFormData()),
            },
            FETCH_TIMEOUT_MS
        );

        const data = await response.json();

        if (response.status === 202 && data.task_id) {
            // Cold start: model is warming up, poll until it's ready
            const result = await pollForResult(data.task_id);
            displayPrice(result.predicted_price);
        } else if (response.status === 200 && data.predicted_price) {
            // Warm path: model already loaded, immediate result
            displayPrice(data.predicted_price);
        } else {
            resultDiv.textContent = 'Error: ' + (data.error || 'Unexpected response');
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.textContent = 'An error occurred. Check the console.';
    }
});

function displayPrice(predictedPrice) {
    if (predictedPrice === undefined || predictedPrice === null) {
        resultDiv.textContent = 'Error: No prediction returned.';
        return;
    }
    const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    }).format(predictedPrice);
    resultDiv.textContent = formattedPrice;
}