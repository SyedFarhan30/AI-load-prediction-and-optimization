// API Base URL
const API_BASE = 'http://localhost:5000';

// Show/Hide Loading Spinner
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

// Create Result Box HTML
function createResultBox(title, message, type = 'success', content = '') {
    let boxClass = `result-box ${type}-box`;
    let html = `
        <div class="${boxClass}">
            <h5>${title}</h5>
            <p>${message}</p>
            ${content}
        </div>
    `;
    return html;
}

// Create Table from Data
function createTableHTML(data, columns) {
    let html = '<table class="table table-sm data-preview-table"><thead><tr>';
    
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            let value = row[col];
            if (typeof value === 'number') {
                value = value.toFixed(2);
            }
            html += `<td>${value}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

// Upload File
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    
    if (!fileInput.files.length) {
        // If no file selected, open file picker instead
        fileInput.click();
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let resultHTML = createResultBox(
                '✓ File Uploaded Successfully',
                `Shape: ${data.shape[0]} rows × ${data.shape[1]} columns`,
                'success',
                `<p><strong>Columns:</strong> ${data.columns.join(', ')}</p>`
            );
            
            document.getElementById('uploadResult').innerHTML = resultHTML;
            setTimeout(() => loadDataPreview(), 500);
        } else {
            let errorMsg = data.error || 'Unknown error';
            document.getElementById('uploadResult').innerHTML = createResultBox(
                '✗ Upload Failed',
                errorMsg,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Upload error:', error);
        let errorHTML = createResultBox(
            '✗ Upload Failed',
            'Error: ' + error.message,
            'error'
        );
        document.getElementById('uploadResult').innerHTML = errorHTML;
    }
}

// Load Data Preview
async function loadDataPreview() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/data_preview`);
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let tableHTML = createTableHTML(data.data, data.columns.slice(0, 6));
            let html = `
                <p><strong>Shape:</strong> ${data.shape[0]} rows × ${data.shape[1]} columns</p>
                <p><strong>Data Types:</strong></p>
                <ul>
                    ${data.columns.slice(0, 5).map(col => `<li>${col}: ${data.dtypes[col] || 'float'}</li>`).join('')}
                </ul>
                ${tableHTML}
            `;
            
            document.getElementById('dataPreviewContainer').innerHTML = html;
        }
    } catch (error) {
        hideLoading();
        console.error('Error loading preview:', error);
    }
}

// Perform EDA
async function performEDA() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/eda`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ EDA Completed',
                'Analysis of data distributions and correlations completed',
                'success'
            );
            
            html += `
                <div class="result-box info-box">
                    <h5>Data Insights</h5>
                    <p><strong>Highest CPU Utilization:</strong> ${data.insights.high_cpu.toFixed(2)}%</p>
                    <p><strong>Average Memory Usage:</strong> ${data.insights.avg_memory.toFixed(2)}%</p>
                    <p><strong>Highest Network Latency:</strong> ${data.insights.high_latency.toFixed(2)} ms</p>
                    <p><strong>Average Power Consumption:</strong> ${data.insights.avg_power.toFixed(2)} W</p>
                </div>
            `;
            
            if (data.plots.distributions) {
                html += `
                    <div class="chart-container">
                        <h5>Feature Distributions</h5>
                        <img src="${data.plots.distributions}" alt="Distributions">
                    </div>
                `;
            }
            
            if (data.plots.correlations) {
                html += `
                    <div class="chart-container">
                        <h5>Feature Correlations</h5>
                        <img src="${data.plots.correlations}" alt="Correlations">
                    </div>
                `;
            }
            
            if (data.plots.time_series) {
                html += `
                    <div class="chart-container">
                        <h5>Time Series Analysis</h5>
                        <img src="${data.plots.time_series}" alt="Time Series">
                    </div>
                `;
            }
            
            document.getElementById('edaResult').innerHTML = html;
        } else {
            document.getElementById('edaResult').innerHTML = createResultBox(
                '✗ EDA Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('EDA error:', error);
    }
}

// Preprocess Data
async function preprocess() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/preprocess`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ Preprocessing Completed',
                `Processed data shape: ${data.processed_shape[0]} rows × ${data.processed_shape[1]} columns`,
                'success',
                `
                    <p><strong>Features Engineered:</strong> ${data.features_engineered}</p>
                    <ul>
                        <li>Missing values handled: ✓</li>
                        <li>Outliers removed: ✓</li>
                        <li>Features normalized: ✓</li>
                        <li>New features created: ${data.statistics.new_features_created}</li>
                    </ul>
                `
            );
            
            document.getElementById('preprocessResult').innerHTML = html;
        } else {
            document.getElementById('preprocessResult').innerHTML = createResultBox(
                '✗ Preprocessing Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Preprocessing error:', error);
    }
}

// Train Model
async function trainModel() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/train_model`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ Model Trained Successfully',
                'Random Forest Regressor trained on preprocessed data',
                'success',
                `
                    <h6>Performance Metrics:</h6>
                    <div style="margin: 15px 0;">
                        <span class="metric-badge">MAE: ${data.metrics.MAE.toFixed(4)}</span>
                        <span class="metric-badge">RMSE: ${data.metrics.RMSE.toFixed(4)}</span>
                        <span class="metric-badge success">R² Score: ${data.metrics.R2_Score.toFixed(4)}</span>
                    </div>
                    <h6 class="mt-3">Top 5 Important Features:</h6>
                    <ul>
                        ${data.top_features.map((f, i) => `<li>${i+1}. ${f.feature}: ${(f.importance * 100).toFixed(2)}%</li>`).join('')}
                    </ul>
                `
            );
            
            document.getElementById('trainResult').innerHTML = html;
        } else {
            document.getElementById('trainResult').innerHTML = createResultBox(
                '✗ Model Training Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Training error:', error);
    }
}

// Make Predictions
async function makePredictions() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ Predictions Completed',
                `Generated predictions for ${data.predictions.length} samples`,
                'success',
                `
                    <h6>Prediction Statistics:</h6>
                    <div style="margin: 15px 0;">
                        <span class="metric-badge">Mean: ${data.statistics.mean_prediction.toFixed(2)}</span>
                        <span class="metric-badge">Std Dev: ${data.statistics.std_prediction.toFixed(2)}</span>
                        <span class="metric-badge">MAE: ${data.statistics.MAE.toFixed(4)}</span>
                        <span class="metric-badge warning">MAPE: ${data.statistics.MAPE.toFixed(2)}%</span>
                    </div>
                    <p><strong>Sample Predictions (first 10):</strong><br>
                    ${data.predictions.slice(0, 10).map(p => p.toFixed(2)).join(', ')}</p>
                `
            );
            
            if (data.plot) {
                html += `
                    <div class="chart-container">
                        <h5>Actual vs Predicted Load</h5>
                        <img src="${data.plot}" alt="Predictions">
                    </div>
                `;
            }
            
            document.getElementById('predictResult').innerHTML = html;
        } else {
            document.getElementById('predictResult').innerHTML = createResultBox(
                '✗ Prediction Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Prediction error:', error);
    }
}

// Detect Anomalies
async function detectAnomalies() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/detect_anomalies`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ Anomaly Detection Completed',
                `Using ${data.summary.method} algorithm`,
                'success',
                `
                    <div style="margin: 15px 0;">
                        <span class="metric-badge danger">${data.summary.total_anomalies} Anomalies Found</span>
                        <span class="metric-badge warning">${data.summary.anomaly_percentage}% of data</span>
                    </div>
                    <h6>Top Anomalies:</h6>
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>CPU (%)</th>
                                <th>Memory (%)</th>
                                <th>Latency (ms)</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.report.top_anomalies.map(a => `
                                <tr>
                                    <td>${a.index}</td>
                                    <td>${a.cpu_utilization.toFixed(2)}</td>
                                    <td>${a.memory_usage.toFixed(2)}</td>
                                    <td>${a.network_latency.toFixed(2)}</td>
                                    <td><strong>${a.anomaly_score.toFixed(3)}</strong></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `
            );
            
            if (data.plot) {
                html += `
                    <div class="chart-container">
                        <h5>Anomalies Visualization</h5>
                        <img src="${data.plot}" alt="Anomalies">
                    </div>
                `;
            }
            
            document.getElementById('anomalyResult').innerHTML = html;
        } else {
            document.getElementById('anomalyResult').innerHTML = createResultBox(
                '✗ Anomaly Detection Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Anomaly detection error:', error);
    }
}

// Run Optimization
async function runOptimization() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/optimize`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            let html = createResultBox(
                '✓ Optimization Completed',
                'Hill Climbing + CSP Scheduling executed successfully',
                'success',
                `
                    <h6>Optimization Results:</h6>
                    <div style="margin: 15px 0;">
                        <span class="metric-badge">Cost: ${data.best_cost.toFixed(2)}</span>
                        <span class="metric-badge success">CPU Reduced to: ${data.best_solution.cpu_utilization.toFixed(2)}%</span>
                        <span class="metric-badge success">Memory: ${data.best_solution.memory_usage.toFixed(2)}%</span>
                        <span class="metric-badge success">Power: ${data.best_solution.power_consumption.toFixed(2)}W</span>
                    </div>
                    
                    <h6 class="mt-3">Scheduling Summary:</h6>
                    <p><strong>Tasks Scheduled:</strong> ${data.scheduling.scheduled_count}/${data.scheduling.total_tasks}</p>
                    <p><strong>Success Rate:</strong> ${data.scheduling.success_rate}%</p>
                    
                    <h6 class="mt-3">Recommendations:</h6>
                    <ul>
                        ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                `
            );
            
            if (data.plot) {
                html += `
                    <div class="chart-container">
                        <h5>Optimization Progress</h5>
                        <img src="${data.plot}" alt="Optimization">
                    </div>
                `;
            }
            
            document.getElementById('optimizeResult').innerHTML = html;
        } else {
            document.getElementById('optimizeResult').innerHTML = createResultBox(
                '✗ Optimization Failed',
                data.error,
                'error'
            );
        }
    } catch (error) {
        hideLoading();
        console.error('Optimization error:', error);
    }
}

// Reset System
async function resetSystem() {
    if (!confirm('Are you sure you want to reset the system? All data will be cleared.')) {
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/reset`, {
            method: 'POST'
        });
        
        hideLoading();
        
        if (response.ok) {
            // Clear all results
            document.getElementById('uploadResult').innerHTML = '';
            document.getElementById('edaResult').innerHTML = '';
            document.getElementById('preprocessResult').innerHTML = '';
            document.getElementById('trainResult').innerHTML = '';
            document.getElementById('predictResult').innerHTML = '';
            document.getElementById('anomalyResult').innerHTML = '';
            document.getElementById('optimizeResult').innerHTML = '';
            document.getElementById('dataPreviewContainer').innerHTML = '<p class="text-muted">Load data to see preview...</p>';
            
            alert('System reset successfully');
        }
    } catch (error) {
        hideLoading();
        console.error('Reset error:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Load Prediction System initialized');
});
