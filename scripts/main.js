// app.js

const form = document.getElementById('prediction-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the form from reloading the page
    const formData = {
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

    // 2. Send the data to your Django API
    fetch('http://127.0.0.1:8000/api/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        // 3. Display the result
        if (data.predicted_price) {
            const formattedPrice = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
            }).format(data.predicted_price);
            resultDiv.textContent = formattedPrice;
        } else {
            resultDiv.textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.textContent = 'An error occurred. Check the console.';
    });
});