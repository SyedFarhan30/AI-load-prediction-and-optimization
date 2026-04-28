// =====================================================================
// CLOUD MONITORING DASHBOARD - ADVANCED JAVASCRIPT
// Modern interactive dashboard with real-time metrics and charts
// =====================================================================

const API_BASE = 'http://localhost:5000';
let charts = {};
let analyticsCharts = {
    prediction: null,
    anomaly: null
};
let metricsData = {
    cpu: [],
    memory: [],
    network: [],
    predictions: []
};
let alerts = [];
let history = [];
let isRealtimeEnabled = true;
let updateInterval = 3000; // 3 seconds

// Analytics and ML Pipeline State
let analyticsState = {
    dataUploaded: false,
    uploadedFileName: null,
    pipelineStatus: {
        eda: { completed: false, error: false },
        preprocessing: { completed: false, error: false },
        training: { completed: false, error: false },
        prediction: { completed: false, error: false },
        anomaly: { completed: false, error: false },
        optimization: { completed: false, error: false }
    }
};

// =====================================================================
// PAGE NAVIGATION
// =====================================================================

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.getAttribute('data-page');
        navigateToPage(page);
    });
});

function navigateToPage(page) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // Show selected page
    const pageElement = document.getElementById(page + '-page');
    if (pageElement) {
        pageElement.style.display = 'block';
    }
    
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-page') === page) {
            item.classList.add('active');
        }
    });
    
    // Initialize page-specific content
    if (page === 'dashboard') {
        initializeDashboard();
    } else if (page === 'analytics') {
        initializeUploadArea();
        updateAnalyticsUploadUI();
    }
}

// =====================================================================
// TIME DISPLAY
// =====================================================================

function updateTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    document.getElementById('timeDisplay').textContent = `${hours}:${minutes}`;
}

setInterval(updateTime, 1000);
updateTime();

// =====================================================================
// DASHBOARD INITIALIZATION
// =====================================================================

function initializeDashboard() {
    if (charts.cpu) return; // Already initialized
    
    // Wait for Chart.js to be available, then initialize
    waitForChart(() => {
        initCharts();
        
        // Start real-time metric simulation
        if (isRealtimeEnabled) {
            simulateMetrics();
            setInterval(simulateMetrics, updateInterval);
        }
        
        updateSystemStatus();
    });
}

// =====================================================================
// CHARTS INITIALIZATION
// =====================================================================

const chartConfig = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
        legend: {
            labels: {
                color: '#cbd5e1',
                font: { size: 12, weight: '500' }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#f1f5f9',
            bodyColor: '#cbd5e1',
            borderColor: '#3b82f6',
            borderWidth: 1
        }
    },
    scales: {
        y: {
            ticks: { color: '#94a3b8', font: { size: 11 } },
            grid: { color: '#334155', drawBorder: false }
        },
        x: {
            ticks: { color: '#94a3b8', font: { size: 11 } },
            grid: { color: '#334155', drawBorder: false }
        }
    }
};

// Wait for Chart.js to be available
function waitForChart(callback, attempts = 0) {
    if (typeof Chart !== 'undefined') {
        callback();
    } else if (attempts < 20) {
        setTimeout(() => waitForChart(callback, attempts + 1), 100);
    } else {
        console.warn('Chart.js did not load from CDN - charts will be disabled');
        // Hide chart containers and show message
        document.querySelectorAll('[id$="Chart"]').forEach(canvas => {
            const container = canvas.parentElement;
            if (container) {
                container.innerHTML = `
                    <div style="padding: 2rem; text-align: center; color: #cbd5e1;">
                        <i class="fas fa-chart-line" style="font-size: 2rem; opacity: 0.5;"></i>
                        <p style="margin-top: 1rem;">Chart visualization unavailable in this environment</p>
                        <small style="color: #94a3b8;">Metrics and alerts are fully functional</small>
                    </div>
                `;
            }
        });
        callback();
    }
}

function initCharts() {
    // Ensure Chart is available
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not available');
        return;
    }
    // CPU Chart
    const cpuCtx = document.getElementById('cpuChart').getContext('2d');
    charts.cpu = new Chart(cpuCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'CPU Usage (%)',
                data: metricsData.cpu.slice(-20) || [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#3b82f6',
                pointHoverRadius: 5
            }]
        },
        options: chartConfig
    });

    // Memory Chart
    const memoryCtx = document.getElementById('memoryChart').getContext('2d');
    charts.memory = new Chart(memoryCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Memory Usage (%)',
                data: metricsData.memory.slice(-20) || [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#10b981',
                pointHoverRadius: 5
            }]
        },
        options: chartConfig
    });

    // Network Chart
    const networkCtx = document.getElementById('networkChart').getContext('2d');
    charts.network = new Chart(networkCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Network Traffic (Mbps)',
                data: metricsData.network.slice(-20) || [],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#f59e0b',
                pointHoverRadius: 5
            }]
        },
        options: chartConfig
    });

    // Prediction Chart
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    charts.prediction = new Chart(predictionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{
                data: [30, 40, 20, 10],
                backgroundColor: [
                    '#10b981',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ],
                borderColor: '#1e293b',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        font: { size: 12 },
                        padding: 15
                    }
                }
            }
        }
    });
}

function generateTimeLabels(count) {
    const labels = [];
    const now = new Date();
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - i * updateInterval);
        labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
    }
    return labels;
}

// =====================================================================
// REAL-TIME METRICS SIMULATION
// =====================================================================

function simulateMetrics() {
    // Fetch real metrics from backend instead of generating random data
    fetchRealMetrics().then(data => {
        if (data) {
            const cpuValue = data.cpu_usage || 50;
            const memoryValue = data.memory_usage || 60;
            const networkValue = data.network_traffic || 50;
            
            // Add to data arrays
            metricsData.cpu.push(cpuValue);
            metricsData.memory.push(memoryValue);
            metricsData.network.push(networkValue);
            
            // Keep only last 60 data points
            metricsData.cpu = metricsData.cpu.slice(-60);
            metricsData.memory = metricsData.memory.slice(-60);
            metricsData.network = metricsData.network.slice(-60);
            
            // Update KPI cards
            updateKPICards(cpuValue, memoryValue, networkValue);
            
            // Update charts
            updateCharts();
            
            // Check for alerts
            checkAndGenerateAlerts(cpuValue, memoryValue, networkValue);
        }
    }).catch(err => {
        console.error('Error fetching metrics:', err);
        // Fallback to default values if fetch fails
        const cpuValue = 50;
        const memoryValue = 60;
        const networkValue = 50;
        
        metricsData.cpu.push(cpuValue);
        metricsData.memory.push(memoryValue);
        metricsData.network.push(networkValue);
        
        metricsData.cpu = metricsData.cpu.slice(-60);
        metricsData.memory = metricsData.memory.slice(-60);
        metricsData.network = metricsData.network.slice(-60);
        
        updateKPICards(cpuValue, memoryValue, networkValue);
        updateCharts();
    });
}

// Fetch real metrics from backend API
async function fetchRealMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            return data;
        } else {
            console.error('Failed to fetch metrics:', response.statusText);
            return null;
        }
    } catch (error) {
        console.error('Error in fetchRealMetrics:', error);
        return null;
    }
}

function updateKPICards(cpu, memory, network) {
    // CPU
    document.getElementById('cpuValue').textContent = Math.round(cpu);
    const cpuStatus = cpu > 80 ? 'danger' : cpu > 60 ? 'warning' : 'success';
    document.getElementById('cpuStatus').innerHTML = `<span class="status-badge ${cpuStatus}">${cpuStatus.toUpperCase()}</span>`;
    
    // Memory
    document.getElementById('memoryValue').textContent = Math.round(memory);
    const memStatus = memory > 80 ? 'danger' : memory > 60 ? 'warning' : 'success';
    document.getElementById('memoryStatus').innerHTML = `<span class="status-badge ${memStatus}">${memStatus.toUpperCase()}</span>`;
    
    // Network
    document.getElementById('networkValue').textContent = Math.round(network);
    const netStatus = network > 80 ? 'danger' : network > 60 ? 'warning' : 'success';
    document.getElementById('networkStatus').innerHTML = `<span class="status-badge ${netStatus}">${netStatus.toUpperCase()}</span>`;
    
    // Predicted Load
    const avgLoad = (cpu + memory) / 2;
    let loadLevel, loadColor;
    if (avgLoad < 40) {
        loadLevel = 'LOW';
        loadColor = 'success';
    } else if (avgLoad < 60) {
        loadLevel = 'MEDIUM';
        loadColor = 'warning';
    } else if (avgLoad < 80) {
        loadLevel = 'HIGH';
        loadColor = 'danger';
    } else {
        loadLevel = 'CRITICAL';
        loadColor = 'danger';
    }
    
    document.getElementById('predictedLoadValue').textContent = loadLevel;
    document.getElementById('loadBadge').textContent = loadLevel;
    document.getElementById('loadBadge').className = `status-badge ${loadColor}`;
}

function updateCharts() {
    if (!charts.cpu) return;
    
    const timeLabels = generateTimeLabels(Math.min(20, metricsData.cpu.length));
    const cpuSlice = metricsData.cpu.slice(-20);
    const memSlice = metricsData.memory.slice(-20);
    const netSlice = metricsData.network.slice(-20);
    
    // Update CPU Chart
    charts.cpu.data.labels = timeLabels;
    charts.cpu.data.datasets[0].data = cpuSlice;
    charts.cpu.update('none');
    
    // Update Memory Chart
    charts.memory.data.labels = timeLabels;
    charts.memory.data.datasets[0].data = memSlice;
    charts.memory.update('none');
    
    // Update Network Chart
    charts.network.data.labels = timeLabels;
    charts.network.data.datasets[0].data = netSlice;
    charts.network.update('none');
}

// =====================================================================
// ALERT SYSTEM
// =====================================================================

function checkAndGenerateAlerts(cpu, memory, network) {
    if (!document.getElementById('autoAlertToggle').checked) return;
    
    const timestamp = new Date().toLocaleTimeString();
    
    if (cpu > 85) {
        addAlert(`High CPU Usage: ${Math.round(cpu)}%`, 'critical', timestamp);
    }
    if (memory > 85) {
        addAlert(`High Memory Usage: ${Math.round(memory)}%`, 'critical', timestamp);
    }
    if (network > 85) {
        addAlert(`High Network Traffic: ${Math.round(network)} Mbps`, 'warning', timestamp);
    }
}

function addAlert(message, severity, timestamp) {
    // Avoid duplicate alerts
    if (alerts.some(a => a.message === message && Date.now() - a.time < 5000)) return;
    
    const alert = {
        id: Date.now(),
        message: message,
        severity: severity,
        time: Date.now(),
        timestamp: timestamp
    };
    
    alerts.unshift(alert);
    if (alerts.length > 20) alerts.pop();
    
    // Update dashboard alerts display
    updateAlertsDisplay();
    
    // Update alert badge
    document.getElementById('alertBadge').textContent = alerts.length;
    
    // Update alerts list on alerts page
    updateAlertsPageDisplay();
}

function updateAlertsPageDisplay() {
    const alertsList = document.getElementById('alertsList');
    if (!alertsList) return;
    
    if (alerts.length === 0) {
        alertsList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-check-circle"></i>
                <p>No alerts. System operating normally.</p>
            </div>
        `;
        return;
    }
    
    const alertsHTML = alerts.map(alert => `
        <div class="card card-dark mt-2">
            <div class="card-body">
                <div class="alert-header">
                    <span class="alert-title">
                        <i class="fas ${alert.severity === 'critical' ? 'fa-exclamation-circle text-danger' : 'fa-info-circle text-warning'}"></i>
                        ${alert.message}
                    </span>
                    <span class="alert-time">${alert.timestamp}</span>
                </div>
                <span class="status-badge ${alert.severity === 'critical' ? 'danger' : 'warning'}" style="margin-top: 0.5rem;">
                    ${alert.severity.toUpperCase()}
                </span>
            </div>
        </div>
    `).join('');
    
    alertsList.innerHTML = alertsHTML;
}

function updateAlertsDisplay() {
    const alertsContainer = document.getElementById('dashboardAlerts');
    
    if (alerts.length === 0) {
        alertsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-check-circle"></i>
                <p>No alerts at the moment. System is operating normally.</p>
            </div>
        `;
        return;
    }
    
    const alertsHTML = alerts.slice(0, 5).map(alert => `
        <div class="alert-item ${alert.severity}">
            <div class="alert-header">
                <span class="alert-title">
                    <i class="fas ${alert.severity === 'critical' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                    ${alert.message}
                </span>
                <span class="alert-time">${alert.timestamp}</span>
            </div>
        </div>
    `).join('');
    
    alertsContainer.innerHTML = alertsHTML;
}

// =====================================================================
// SYSTEM STATUS
// =====================================================================

function updateSystemStatus() {
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('systemStatusText');
    
    // Randomly determine status for demo
    const isHealthy = Math.random() > 0.2;
    
    if (isHealthy) {
        statusDot.style.background = '#10b981';
        statusText.textContent = 'System Status: Healthy';
    } else {
        statusDot.style.background = '#f59e0b';
        statusText.textContent = 'System Status: Warning';
    }
}

// =====================================================================
// UPLOAD FUNCTIONALITY
// =====================================================================

function initializeUploadArea() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const mainFileInput = document.getElementById('mainFileInput');
    const mainUploadArea = document.getElementById('mainUploadArea');
    
    // Analytics page upload
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', (e) => handleFileDrop(e, fileInput));
        uploadArea.addEventListener('click', () => fileInput.click());
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) uploadFileDirectly(file);
        });
    }
    
    // Upload page upload
    if (mainUploadArea) {
        mainUploadArea.addEventListener('dragover', handleDragOver);
        mainUploadArea.addEventListener('dragleave', handleDragLeave);
        mainUploadArea.addEventListener('drop', (e) => handleFileDrop(e, mainFileInput));
        mainUploadArea.addEventListener('click', () => mainFileInput.click());
    }
    
    if (mainFileInput) {
        mainFileInput.addEventListener('change', () => {
            const file = mainFileInput.files[0];
            if (file) uploadFileDirectly(file);
        });
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.style.background = 'rgba(59, 130, 246, 0.2)';
    e.target.style.borderColor = '#3b82f6';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.style.background = '';
    e.target.style.borderColor = '';
}

function handleFileDrop(e, fileInput) {
    e.preventDefault();
    e.stopPropagation();
    e.target.style.background = '';
    e.target.style.borderColor = '';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.name.toLowerCase().endsWith('.csv')) {
            uploadFileDirectly(file);
        } else {
            showToast('Please drop a CSV file', 'error');
        }
    }
}

// Upload file directly from drag-drop or file input
async function uploadFileDirectly(file) {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            // Update analytics state
            analyticsState.dataUploaded = true;
            analyticsState.uploadedFileName = file.name;
            analyticsState.pipelineStatus = {
                eda: { completed: false, error: false },
                preprocessing: { completed: false, error: false },
                training: { completed: false, error: false },
                prediction: { completed: false, error: false },
                anomaly: { completed: false, error: false },
                optimization: { completed: false, error: false }
            };
            
            // Update analytics page UI
            updateAnalyticsUploadUI();
            
            showToast(`✓ File uploaded: ${data.shape[0]} rows, ${data.shape[1]} columns`, 'success');
            document.getElementById('mainUploadStatus').innerHTML = `
                <div class="alert alert-success" role="alert">
                    <i class="fas fa-check-circle"></i>
                    <strong>Upload Successful!</strong> Dataset loaded with ${data.shape[0]} rows and ${data.shape[1]} columns.
                </div>
            `;
            showDataPreview(data, 'mainDataPreview');
        } else {
            showToast(`✗ Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showLoading(false);
        showToast(`✗ Upload error: ${error.message}`, 'error');
        console.error('Upload error details:', error);
    }
}

async function uploadFile(fileInput, isMainUpload = false) {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            showToast(`File uploaded successfully! ${data.shape[0]} rows, ${data.shape[1]} columns`, 'success');
            if (isMainUpload) {
                document.getElementById('mainUploadStatus').innerHTML = `
                    <div class="alert alert-success" role="alert">
                        <i class="fas fa-check-circle"></i>
                        <strong>Upload Successful!</strong> Dataset loaded with ${data.shape[0]} rows and ${data.shape[1]} columns.
                    </div>
                `;
                showDataPreview(data, 'mainDataPreview');
            }
        } else {
            showToast(`Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showLoading(false);
        showToast(`Upload error: ${error.message}`, 'error');
    }
}

function showDataPreview(data, containerId) {
    if (!data.preview || data.preview.length === 0) return;
    
    const container = document.getElementById(containerId);
    const columns = data.columns.slice(0, 6);
    const rows = data.preview;
    
    let table = '<table class="table table-dark table-sm"><thead><tr>';
    columns.forEach(col => table += `<th>${col}</th>`);
    table += '</tr></thead><tbody>';
    
    rows.forEach(row => {
        table += '<tr>';
        columns.forEach(col => {
            const value = row[col] && typeof row[col] === 'number' ? row[col].toFixed(2) : row[col];
            table += `<td>${value}</td>`;
        });
        table += '</tr>';
    });
    
    table += '</tbody></table>';
    container.innerHTML = table;
}

// =====================================================================
// ANALYTICS PAGE FUNCTIONS
// =====================================================================

function updateAnalyticsUploadUI() {
    const uploadCard = document.getElementById('uploadCard');
    const uploadStatusCard = document.getElementById('uploadStatusCard');
    const pipelineDisabledMsg = document.getElementById('pipelineDisabledMessage');
    const pipelineSteps = document.getElementById('pipelineSteps');
    
    // Add safety checks
    if (!uploadCard || !uploadStatusCard || !pipelineDisabledMsg || !pipelineSteps) {
        console.warn('Analytics page elements not found');
        return;
    }
    
    if (analyticsState.dataUploaded) {
        // Hide upload card, show status card
        uploadCard.style.display = 'none';
        uploadStatusCard.style.display = 'block';
        const fileNameElement = document.getElementById('uploadedFileName');
        if (fileNameElement) {
            fileNameElement.textContent = analyticsState.uploadedFileName;
        }
        
        // Show pipeline
        pipelineDisabledMsg.style.display = 'none';
        pipelineSteps.style.display = 'grid';
    } else {
        // Show upload card, hide status card
        uploadCard.style.display = 'block';
        uploadStatusCard.style.display = 'none';
        pipelineDisabledMsg.style.display = 'block';
        pipelineSteps.style.display = 'none';
        
        // Clear results
        const resultsElement = document.getElementById('analyticsResults');
        if (resultsElement) {
            resultsElement.innerHTML = '';
        }
    }
}

function resetAnalytics() {
    analyticsState.dataUploaded = false;
    analyticsState.uploadedFileName = null;
    analyticsState.pipelineStatus = {
        eda: { completed: false, error: false },
        preprocessing: { completed: false, error: false },
        training: { completed: false, error: false },
        prediction: { completed: false, error: false },
        anomaly: { completed: false, error: false },
        optimization: { completed: false, error: false }
    };
    
    // Reset status badges
    document.getElementById('edaStatus').innerHTML = '';
    document.getElementById('preprocessStatus').innerHTML = '';
    document.getElementById('trainingStatus').innerHTML = '';
    document.getElementById('predictionStatus').innerHTML = '';
    document.getElementById('anomalyStatus').innerHTML = '';
    document.getElementById('optimizationStatus').innerHTML = '';
    
    updateAnalyticsUploadUI();
    showToast('Analytics reset. Please upload a new file.', 'info');
}

function updatePipelineStatus(stepName, status) {
    // Map step names to status element IDs
    const idMap = {
        'preprocessing': 'prep-status-span',
        'training': 'trainingStatus',
        'eda': 'edaStatus',
        'prediction': 'predictionStatus',
        'anomaly': 'anomalyStatus',
        'optimization': 'optimizationStatus'
    };
    
    const statusElement = document.getElementById(idMap[stepName] || stepName + 'Status');
    if (!statusElement) return;
    
    if (status === 'running') {
        statusElement.innerHTML = '<span style="color: #f59e0b; margin-left: 10px;"><i class="fas fa-spinner fa-spin"></i> Processing...</span>';
    } else if (status === 'completed') {
        statusElement.innerHTML = '<span style="color: #10b981; margin-left: 10px;"><i class="fas fa-check-circle"></i> Completed</span>';
        analyticsState.pipelineStatus[stepName].completed = true;
        analyticsState.pipelineStatus[stepName].error = false;
    } else if (status === 'error') {
        statusElement.innerHTML = '<span style="color: #ef4444; margin-left: 10px;"><i class="fas fa-times-circle"></i> Error</span>';
        analyticsState.pipelineStatus[stepName].error = true;
    }
}

// =====================================================================
// ML PIPELINE FUNCTIONS
// =====================================================================

async function performEDA() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    updatePipelineStatus('eda', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/eda`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('eda', 'completed');
            showToast('✓ EDA Completed Successfully!', 'success');
            
            let html = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-pie" style="color: #3b82f6;"></i> Exploratory Data Analysis Results</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Key Insights</h6>
                                <p><strong>Highest CPU:</strong> ${data.insights.high_cpu.toFixed(2)}%</p>
                                <p><strong>Avg Memory:</strong> ${data.insights.avg_memory.toFixed(2)}%</p>
                                <p><strong>Highest Latency:</strong> ${data.insights.high_latency.toFixed(2)} ms</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Data Summary</h6>
                                <p><strong>Total Records:</strong> ${data.records_count || 'N/A'}</p>
                                <p><strong>Features:</strong> ${data.features_count || 'N/A'}</p>
                                <p><strong>Missing Values:</strong> ${data.missing_values || '0'}</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = html;
        } else {
            updatePipelineStatus('eda', 'error');
            showToast(`✗ EDA Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('eda', 'error');
        showLoading(false);
        showToast(`✗ EDA Error: ${error.message}`, 'error');
    }
}

async function preprocess() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    updatePipelineStatus('preprocessing', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/preprocess`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('preprocessing', 'completed');
            showToast('✓ Preprocessing Completed!', 'success');
            
            let html = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-cogs" style="color: #10b981;"></i> Data Preprocessing Results</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Processing Summary</h6>
                                <p><strong>Original Shape:</strong> (${data.original_shape?.[0]}, ${data.original_shape?.[1]})</p>
                                <p><strong>Processed Shape:</strong> (${data.processed_shape[0]}, ${data.processed_shape[1]})</p>
                                <p><strong>Features Created:</strong> ${data.statistics.new_features_created}</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Transformations Applied</h6>
                                <p><i class="fas fa-check" style="color: #10b981;"></i> Missing values handled</p>
                                <p><i class="fas fa-check" style="color: #10b981;"></i> Outliers handled</p>
                                <p><i class="fas fa-check" style="color: #10b981;"></i> Features normalized</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = html;
        } else {
            updatePipelineStatus('preprocessing', 'error');
            showToast(`✗ Preprocessing Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('preprocessing', 'error');
        showLoading(false);
        showToast(`✗ Preprocessing Error: ${error.message}`, 'error');
    }
}

async function trainModel() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    if (!analyticsState.pipelineStatus.preprocessing.completed) {
        showToast('Please run Preprocessing first', 'warning');
        return;
    }
    
    updatePipelineStatus('training', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/train_model`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('training', 'completed');
            showToast('✓ Model Trained Successfully!', 'success');
            
            let html = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-brain" style="color: #8b5cf6;"></i> Model Training Results</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Model Performance</h6>
                                <p><strong>Model Type:</strong> Random Forest</p>
                                <p><strong>MAE:</strong> ${data.metrics.MAE?.toFixed(4) || 'N/A'}</p>
                                <p><strong>RMSE:</strong> ${data.metrics.RMSE?.toFixed(4) || 'N/A'}</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Accuracy Score</h6>
                                <div style="background: linear-gradient(90deg, #1e293b 0%, #3b82f6 ${(data.metrics.R2_Score * 100).toFixed(1)}%); height: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                    R² Score: ${data.metrics.R2_Score?.toFixed(4) || 'N/A'}
                                </div>
                            </div>
                        </div>
                        <hr style="border-color: #334155;">
                        <h6>Top Features</h6>
                        <div style="max-height: 200px; overflow-y: auto;">
                            ${data.top_features?.slice(0, 5).map((f, i) => `
                                <p style="margin: 5px 0;"><span style="color: #f59e0b;">#${i+1}</span> ${f.feature}: ${f.importance?.toFixed(4) || 'N/A'}</p>
                            `).join('') || '<p>No feature importance data</p>'}
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = html;
        } else {
            updatePipelineStatus('training', 'error');
            showToast(`✗ Training Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('training', 'error');
        showLoading(false);
        showToast(`✗ Training Error: ${error.message}`, 'error');
    }
}

async function makePredictions() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    if (!analyticsState.pipelineStatus.training.completed) {
        showToast('Please train the model first', 'warning');
        return;
    }
    
    updatePipelineStatus('prediction', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/predict`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('prediction', 'completed');
            showToast('✓ Predictions Generated!', 'success');

            const cpuStats = data.statistics?.cpu || {};
            const memoryStats = data.statistics?.memory || {};
            const cpuPreds = data.predictions?.cpu || [];
            const memoryPreds = data.predictions?.memory || [];
            const labels = data.labels || cpuPreds.map((_, idx) => `Interval ${idx + 1}`);
            const currentCpu = data.current_predicted_usage?.cpu;
            const currentMemory = data.current_predicted_usage?.memory;
            const cpuForecast = data.forecast?.cpu?.values || [];
            const memoryForecast = data.forecast?.memory?.values || [];
            const cpuTrend = data.forecast?.cpu?.trend || 'stable';
            const memoryTrend = data.forecast?.memory?.trend || 'stable';

            const resultHTML = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line" style="color: #f59e0b;"></i> Load Prediction Insights</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <strong>Current Predicted Usage:</strong>
                            CPU ${currentCpu !== null && currentCpu !== undefined ? currentCpu.toFixed(2) : 'N/A'}% |
                            Memory ${currentMemory !== null && currentMemory !== undefined ? currentMemory.toFixed(2) : 'N/A'}%
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h6>CPU Usage Forecast</h6>
                                <p><strong>Trend:</strong> ${cpuTrend}</p>
                                <p><strong>Current:</strong> ${currentCpu !== null && currentCpu !== undefined ? currentCpu.toFixed(2) : 'N/A'}%</p>
                                <p><strong>Range:</strong> ${cpuPreds.length ? Math.min(...cpuPreds).toFixed(2) : 'N/A'} - ${cpuPreds.length ? Math.max(...cpuPreds).toFixed(2) : 'N/A'}%</p>
                                <p class="text-muted small">MAE: ${cpuStats.MAE?.toFixed(3) || 'N/A'}</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Memory Usage Forecast</h6>
                                <p><strong>Trend:</strong> ${memoryTrend}</p>
                                <p><strong>Current:</strong> ${currentMemory !== null && currentMemory !== undefined ? currentMemory.toFixed(2) : 'N/A'}%</p>
                                <p><strong>Range:</strong> ${memoryPreds.length ? Math.min(...memoryPreds).toFixed(2) : 'N/A'} - ${memoryPreds.length ? Math.max(...memoryPreds).toFixed(2) : 'N/A'}%</p>
                                <p class="text-muted small">MAE: ${memoryStats?.MAE?.toFixed(3) || 'N/A'}</p>
                            </div>
                        </div>
                        <hr style="border-color: #334155;">
                        <p class="mb-2"><strong>Short-term Forecast:</strong> Next ${cpuForecast.length || 0} intervals</p>
                        <canvas id="predictionInsightChart" height="100"></canvas>
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = resultHTML;
            renderAnalyticsPredictionChart(labels, cpuPreds, memoryPreds, cpuForecast, memoryForecast);
            addToHistory('Prediction', 'completed', false, 0.87);
        } else {
            updatePipelineStatus('prediction', 'error');
            showToast(`✗ Prediction Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('prediction', 'error');
        showLoading(false);
        showToast(`✗ Prediction Error: ${error.message}`, 'error');
    }
}

async function detectAnomalies() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    if (!analyticsState.pipelineStatus.preprocessing.completed) {
        showToast('Please run Preprocessing first', 'warning');
        return;
    }
    
    updatePipelineStatus('anomaly', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/detect_anomalies`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('anomaly', 'completed');
            showToast(`✓ Anomalies Detected: ${data.summary?.total_anomalies || 0}`, 'success');
            
            const anomalyPercentage = data.summary?.anomaly_percentage || 0;
            const cpuAnomalies = data.anomaly_details?.cpu || [];
            const memoryAnomalies = data.anomaly_details?.memory || [];
            const merged = data.anomaly_details?.merged || [];
            const labels = data.series?.labels || [];
            const cpuSeries = data.series?.cpu || [];
            const memorySeries = data.series?.memory || [];

            const topNarratives = merged.slice(0, 5).map(item => `
                <li><strong>${item.location}</strong>: ${item.explanation} (value: ${item.value.toFixed(2)}%)</li>
            `).join('');

            const resultHTML = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i> Anomaly Insights</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Detection Summary</h6>
                                <p><strong>Total Anomalies:</strong> ${data.summary?.total_anomalies || 0}</p>
                                <p><strong>CPU Anomalies:</strong> ${data.summary?.cpu_anomalies || 0}</p>
                                <p><strong>Memory Anomalies:</strong> ${data.summary?.memory_anomalies || 0}</p>
                                <p><strong>Anomaly %:</strong> ${anomalyPercentage.toFixed(2)}%</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Anomaly Distribution</h6>
                                <div style="background: linear-gradient(90deg, #10b981 0%, #ef4444 ${anomalyPercentage}%); height: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">
                                    ${anomalyPercentage.toFixed(2)}% Anomalous
                                </div>
                                <p class="mt-2 mb-0 small text-muted">Red markers: CPU spikes | Yellow markers: Memory spikes</p>
                            </div>
                        </div>
                        <hr style="border-color: #334155;">
                        <canvas id="anomalyInsightChart" height="95"></canvas>
                        ${merged.length ? `
                            <hr style="border-color: #334155;">
                            <h6>Actionable Explanations</h6>
                            <ul class="small mb-0">${topNarratives}</ul>
                        ` : ''}
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = resultHTML;
            renderAnalyticsAnomalyChart(labels, cpuSeries, memorySeries, cpuAnomalies, memoryAnomalies);
        } else {
            updatePipelineStatus('anomaly', 'error');
            showToast(`✗ Anomaly Detection Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('anomaly', 'error');
        showLoading(false);
        showToast(`✗ Anomaly Detection Error: ${error.message}`, 'error');
    }
}

async function runOptimization() {
    if (!analyticsState.dataUploaded) {
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    if (!analyticsState.pipelineStatus.training.completed) {
        showToast('Please train the model first', 'warning');
        return;
    }
    
    updatePipelineStatus('optimization', 'running');
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/optimize`, { method: 'POST' });
        const data = await response.json();
        showLoading(false);
        
        if (response.ok) {
            updatePipelineStatus('optimization', 'completed');
            showToast('✓ Optimization Complete!', 'success');

            const currentCpu = data.optimization?.current_cpu_usage;
            const currentMemory = data.optimization?.current_memory_usage;
            const optimizedCpu = data.optimization?.optimized_cpu_usage;
            const optimizedMemory = data.optimization?.optimized_memory_usage;
            const hasMemoryUsage = currentMemory !== null && currentMemory !== undefined;
            
            const resultHTML = `
                <div class="card card-dark mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-leaf" style="color: #10b981;"></i> Optimization Insights</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Usage Comparison</h6>
                                <p><strong>Current CPU Usage:</strong> ${currentCpu?.toFixed(2) || 'N/A'}%</p>
                                <p><strong>Optimized CPU Usage:</strong> ${optimizedCpu?.toFixed(2) || 'N/A'}%</p>
                                <p><strong>Current Memory Usage:</strong> ${hasMemoryUsage ? currentMemory.toFixed(2) : 'N/A'}%</p>
                                <p><strong>Optimized Memory Usage:</strong> ${optimizedMemory !== null && optimizedMemory !== undefined ? optimizedMemory.toFixed(2) : 'N/A'}%</p>
                                <p><strong>Optimized Usage:</strong> ${data.optimization?.optimized_usage?.toFixed(2) || 'N/A'}%</p>
                                <p><strong>Potential Savings:</strong> ${data.optimization?.potential_savings?.toFixed(2) || 'N/A'}%</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Actionable Recommendations</h6>
                                <p><i class="fas fa-lightbulb" style="color: #f59e0b;"></i> CPU limit range: 75-85% (recommended ${data.optimization?.cpu_limit || 'N/A'}%).</p>
                                <p><i class="fas fa-lightbulb" style="color: #f59e0b;"></i> Memory limit range: 80-90% (recommended ${data.optimization?.memory_limit || 'N/A'}%).</p>
                                <p class="small text-muted mb-0">Apply these thresholds to reduce spike risk while maintaining throughput.</p>
                            </div>
                        </div>
                        ${data.scheduling ? `
                            <hr style="border-color: #334155;">
                            <h6>Scheduling Suggestion</h6>
                            <p><strong>Strategy:</strong> ${data.scheduling?.strategy || 'Optimal'}</p>
                            <p><strong>Tasks Scheduled:</strong> ${data.scheduling?.tasks_count || 0}</p>
                            <p><strong>Recommendation:</strong> ${data.scheduling?.suggestion || 'Gradually spread high-load tasks across intervals.'}</p>
                        ` : ''}
                    </div>
                </div>
            `;
            document.getElementById('analyticsResults').innerHTML = resultHTML;
        } else {
            updatePipelineStatus('optimization', 'error');
            showToast(`✗ Optimization Error: ${data.error}`, 'error');
        }
    } catch (error) {
        updatePipelineStatus('optimization', 'error');
        showLoading(false);
        showToast(`✗ Optimization Error: ${error.message}`, 'error');
    }
}

// =====================================================================
// HISTORY TRACKING
// =====================================================================

function addToHistory(loadLevel, anomaly, confidence) {
    const timestamp = new Date().toLocaleTimeString();
    history.unshift({
        time: timestamp,
        load: loadLevel,
        anomaly: anomaly,
        confidence: confidence
    });
    
    if (history.length > 50) history.pop();
    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    const table = document.getElementById('historyTable');
    if (history.length === 0) {
        table.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No predictions yet.</td></tr>';
        return;
    }
    
    const rows = history.slice(0, 20).map(item => `
        <tr>
            <td>${item.time}</td>
            <td><span class="status-badge">${item.load}</span></td>
            <td>${item.anomaly ? '<span class="badge bg-danger">Yes</span>' : '<span class="badge bg-success">No</span>'}</td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
            <td><i class="fas fa-check-circle text-success"></i> Logged</td>
        </tr>
    `).join('');
    
    table.innerHTML = rows;
}

// =====================================================================
// SETTINGS
// =====================================================================

function saveSettings() {
    isRealtimeEnabled = document.getElementById('realtimeToggle').checked;
    updateInterval = parseInt(document.getElementById('updateInterval').value) * 1000;
    showToast('Settings Saved!', 'success');
}

function renderAnalyticsPredictionChart(labels, cpuSeries, memorySeries, cpuForecast, memoryForecast) {
    const canvas = document.getElementById('predictionInsightChart');
    if (!canvas || typeof Chart === 'undefined') return;

    if (analyticsCharts.prediction) {
        analyticsCharts.prediction.destroy();
    }

    const forecastLabels = Array.from({ length: cpuForecast.length || 0 }, (_, i) => `F${i + 1}`);
    const chartLabels = [...labels, ...forecastLabels];
    const cpuData = [...cpuSeries, ...cpuForecast];
    const hasMemory = memorySeries.length > 0 || memoryForecast.length > 0;
    const memoryData = hasMemory ? [...memorySeries, ...memoryForecast] : [];
    const splitIndex = labels.length;

    const datasets = [
        {
            label: 'CPU Usage (%)',
            data: cpuData,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249, 115, 22, 0.12)',
            borderWidth: 2,
            tension: 0.35,
            pointRadius: cpuData.map((_, i) => (i >= splitIndex ? 0 : 2)),
            fill: false
        }
    ];

    if (hasMemory) {
        datasets.push({
            label: 'Memory Usage (%)',
            data: memoryData,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            borderWidth: 2,
            tension: 0.35,
            pointRadius: memoryData.map((_, i) => (i >= splitIndex ? 0 : 2)),
            fill: false
        });
    }

    analyticsCharts.prediction = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: chartLabels, datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#cbd5e1' } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: {
                    suggestedMin: 0,
                    suggestedMax: 100,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 12 },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

function renderAnalyticsAnomalyChart(labels, cpuSeries, memorySeries, cpuAnomalies, memoryAnomalies) {
    const canvas = document.getElementById('anomalyInsightChart');
    if (!canvas || typeof Chart === 'undefined') return;

    if (analyticsCharts.anomaly) {
        analyticsCharts.anomaly.destroy();
    }

    const cpuAnomalyData = labels.map(() => null);
    const memoryAnomalyData = labels.map(() => null);

    cpuAnomalies.forEach(item => {
        if (item.index >= 0 && item.index < cpuAnomalyData.length) {
            cpuAnomalyData[item.index] = item.value;
        }
    });
    memoryAnomalies.forEach(item => {
        if (item.index >= 0 && item.index < memoryAnomalyData.length) {
            memoryAnomalyData[item.index] = item.value;
        }
    });

    const datasets = [
        {
            label: 'CPU Usage (%)',
            data: cpuSeries,
            borderColor: '#f97316',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35
        },
        {
            type: 'scatter',
            label: 'CPU Anomalies',
            data: cpuAnomalyData,
            borderColor: '#ef4444',
            backgroundColor: '#ef4444',
            pointRadius: 5,
            showLine: false
        }
    ];

    if (memorySeries.length > 0) {
        datasets.push({
            label: 'Memory Usage (%)',
            data: memorySeries,
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35
        });
        datasets.push({
            type: 'scatter',
            label: 'Memory Anomalies',
            data: memoryAnomalyData,
            borderColor: '#fde047',
            backgroundColor: '#fde047',
            pointRadius: 5,
            showLine: false
        });
    }

    analyticsCharts.anomaly = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#cbd5e1' } },
                tooltip: { mode: 'nearest', intersect: true }
            },
            scales: {
                y: {
                    suggestedMin: 0,
                    suggestedMax: 100,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 12 },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

// =====================================================================
// UI UTILITIES
// =====================================================================

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.classList.add('show');
    } else {
        spinner.classList.remove('show');
    }
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// =====================================================================
// INITIALIZATION
// =====================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize page
    navigateToPage('dashboard');
    initializeDashboard();
    initializeUploadArea();
    
    // Sidebar toggle for mobile
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }
});

// Show toast on page load
window.addEventListener('load', () => {
    showToast('Welcome to Cloud Monitoring Dashboard!', 'success');
});
