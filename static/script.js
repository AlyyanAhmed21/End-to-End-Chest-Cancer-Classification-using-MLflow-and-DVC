document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Elements ---
    const fileInput = document.getElementById('fileInput');
    const uploadLabel = document.querySelector('.upload-label');
    const imagePreviewContainer = document.querySelector('.image-preview-container');
    const imagePreview = document.getElementById('imagePreview');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('result-container');
    const jsonResponse = document.getElementById('jsonResponse').querySelector('code');

    let base64Image = null;

    // --- Event Listeners ---
    fileInput.addEventListener('change', handleFileSelect);
    removeImageBtn.addEventListener('click', resetUploader);
    predictBtn.addEventListener('click', handlePrediction);

    // --- Functions ---

    /**
     * Handles the file selection, reads the file as a Base64 string,
     * and updates the UI to show the preview.
     */
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Display the image preview
                imagePreview.src = e.target.result;
                uploadLabel.style.display = 'none';
                imagePreviewContainer.style.display = 'block';

                // Store the Base64 string (without the data URI prefix)
                base64Image = e.target.result.split(',')[1];
                
                // Enable the predict button
                predictBtn.disabled = false;
                resultContainer.innerHTML = '<p class="text-muted">Ready to predict.</p>';
                jsonResponse.textContent = 'Waiting for response...';
            };
            reader.readAsDataURL(file);
        }
    }

    /**
     * Resets the uploader to its initial state.
     */
    function resetUploader() {
        fileInput.value = ''; // Clear the file input
        base64Image = null;
        imagePreview.src = '#';
        uploadLabel.style.display = 'flex';
        imagePreviewContainer.style.display = 'none';
        predictBtn.disabled = true;
        resultContainer.innerHTML = '<p class="text-muted">Results will be displayed here after prediction.</p>';
        jsonResponse.textContent = 'Waiting for response...';
    }

    /**
     * Handles the prediction API call.
     */
    async function handlePrediction() {
        if (!base64Image) {
            alert('Please upload an image first.');
            return;
        }

        setLoadingState(true);

        // !! IMPORTANT: Change this URL to your actual API endpoint !!
        const apiUrl = '/predict'; // Example for a local Flask app

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Prediction Error:', error);
            displayError(error.message);
        } finally {
            setLoadingState(false);
        }
    }

    /**
     * Displays the prediction results in a user-friendly format.
     */
    function displayResults(data) {
        // Assuming the response is like: [{"prediction": "Normal"}]
        const prediction = data[0]?.prediction; // Safely access the prediction
        
        let resultHtml = '';
        if (prediction) {
            if (prediction.toLowerCase() === 'normal') {
                resultHtml = `
                    <div class="result-normal">
                        <i class="fas fa-check-circle result-icon"></i>
                        <h3>Prediction: Normal</h3>
                        <p>The model predicts that the scan is not cancerous.</p>
                    </div>`;
            } else {
                resultHtml = `
                    <div class="result-cancer">
                        <i class="fas fa-exclamation-triangle result-icon"></i>
                        <h3>Prediction: Cancer Detected</h3>
                        <p>The model predicts a high probability of malignancy. Please consult a medical professional.</p>
                    </div>`;
            }
        } else {
             resultHtml = `<p>Could not determine prediction from the response.</p>`;
        }
        
        resultContainer.innerHTML = resultHtml;
        jsonResponse.textContent = JSON.stringify(data, null, 2);
    }
    
    /**
     * Displays an error message in the UI.
     */
    function displayError(errorMessage) {
        resultContainer.innerHTML = `
            <div class="text-danger">
                <i class="fas fa-times-circle result-icon"></i>
                <h3>Prediction Failed</h3>
                <p>${errorMessage}</p>
            </div>`;
        jsonResponse.textContent = `Error: ${errorMessage}`;
    }

    /**
     * Manages the loading state of the predict button.
     */
    function setLoadingState(isLoading) {
        const spinner = predictBtn.querySelector('.spinner-border');
        const btnText = predictBtn.querySelector('.btn-text');

        if (isLoading) {
            predictBtn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.style.display = 'none';
        } else {
            predictBtn.disabled = false;
            spinner.style.display = 'none';
            btnText.style.display = 'inline-block';
        }
    }
});