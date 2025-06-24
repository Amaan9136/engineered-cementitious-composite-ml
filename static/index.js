// Form validation and utilities
function resetForm() {
    document.getElementById('prediction-form').reset();
}

// Train models functionality
async function trainModels() {
    const datasetSelect = document.getElementById('dataset-select');
    const dataset = datasetSelect.value;

    if (!dataset) {
        alert('Please select a dataset');
        return;
    }

    const resultsDiv = document.getElementById('training-results');
    resultsDiv.innerHTML = '<div class="loader"></div><p style="text-align: center; color: white; margin-top: 10px;">Training models...</p>';

    const formData = new FormData();
    formData.append('dataset', dataset);

    try {
        const response = await fetch('/train_models', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            resultsDiv.innerHTML = `
                        <div class="success-message">
                            <h3>✅ Models trained successfully!</h3>
                            <p>All regression models have been trained on <strong>${result.dataset}</strong></p>
                            <img src="${result.comparison_plot}" alt="Model Comparison" style="max-width: 100%; margin-top: 20px; border-radius: 10px;">
                        </div>
                    `;
        } else {
            resultsDiv.innerHTML = `
                        <div class="error-message">
                            <h3>❌ Error</h3>
                            <p>${result.error}</p>
                        </div>
                    `;
        }
    } catch (error) {
        resultsDiv.innerHTML = `
                    <div class="error-message">
                        <h3>❌ Error</h3>
                        <p>Failed to train models: ${error.message}</p>
                    </div>
                `;
    }
}

// Add input validation
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');

    if (form) {
        form.addEventListener('submit', function (e) {
            const inputs = form.querySelectorAll('input[type="number"]');
            let valid = true;

            inputs.forEach(input => {
                if (input.value === '' || isNaN(input.value)) {
                    if (input.name !== 'experimental') { // experimental is optional
                        valid = false;
                        input.style.borderColor = '#ff6b6b';
                    }
                } else {
                    input.style.borderColor = '#e0e0e0';
                }
            });

            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required numeric fields with valid values');
            }
        });
    }

    // Add real-time validation
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function () {
            if (this.value === '' || isNaN(this.value)) {
                this.style.borderColor = '#ff6b6b';
            } else {
                this.style.borderColor = '#667eea';
            }
        });
    });
});