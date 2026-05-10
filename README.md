# QUICK START GUIDE

## Installation & Running in 5 Minutes

### Prerequisites:
- Python 3.7 or higher
- Windows/Mac/Linux

### Step 1: Install Python Packages
```bash

pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed Flask-2.3.0
Successfully installed pandas-1.5.3
Successfully installed numpy-1.24.3
... (other packages)
```

### Step 2: Verify Dataset
Make sure `Big_data_dataset.csv` is in the project folder:
```bash
dir "Big_data_dataset.csv"  # Windows
ls Big_data_dataset.csv      # Mac/Linux
```

### Step 3: Run Flask Application
```bash
python app.py
```

**You should see:**
```
WARNING in app.run() is not intended for production use!
Running on http://127.0.0.1:5000
```

### Step 4: Open Web Browser
Go to: **http://localhost:5000**

### Step 5: Use the System

Follow this order:

1. **Upload Data**
   - Click "📤 Upload Data" tab
   - Select `Big_data_dataset.csv`
   - Click "Upload & Load"
   - Wait for preview

2. **Explore Data (Optional)**
   - Click "📊 EDA" tab
   - Click "Run EDA" button
   - View distribution plots and correlations

3. **Preprocess**
   - Click "🔧 Preprocessing" tab
   - Click "Run Preprocessing"
   - Wait for completion

4. **Train Model**
   - Click "🧠 Train Model" tab
   - Click "Train Model"
   - View MAE, RMSE, R² Score

5. **Make Predictions**
   - Click "📈 Predictions" tab
   - Click "Make Predictions"
   - View prediction graph

6. **Detect Anomalies**
   - Click "⚠️ Anomalies" tab
   - Click "Detect Anomalies"
   - View anomaly visualization

7. **Optimize Resources**
   - Click "⚡ Optimization" tab
   - Click "Run Optimization"
   - View optimization results and recommendations

8. **View Results**
   - Each tab displays results with graphs and statistics
   - Click "Reset System" to start over

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Run `pip install -r requirements.txt`

### Error: "Address already in use"
**Solution:** 
```bash
# Change port in app.py, last line:
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead
```

### Error: "No such file or directory: 'Big_data_dataset.csv'"
**Solution:** Make sure CSV file is in the project root directory

### Prediction takes too long
**Solution:** This is normal for first-time training. Wait 30-60 seconds.

### Graphs not showing
**Solution:** Make sure matplotlib is installed:
```bash
pip install --upgrade matplotlib
```

---

## File Explanations

### Core Modules:
- `data_preprocessing.py` - Cleans data, handles missing values, engineers features
- `load_prediction_model.py` - Random Forest model for CPU prediction
- `anomaly_detection.py` - Isolation Forest for anomaly detection
- `optimization_module.py` - Hill Climbing algorithm + CSP scheduler
- `eda_module.py` - Data visualization and statistical analysis

### Web Application:
- `app.py` - Flask backend with all API endpoints
- `templates/index.html` - Web interface (HTML)
- `static/css/style.css` - Styling and layout
- `static/js/app.js` - Frontend interactions (JavaScript)

### Documentation:
- `DOCUMENTATION.md` - Complete guide (for VIVA exam)
- `README.md` - Quick start (this file)
- `requirements.txt` - Python dependencies

---

## What Each Button Does

### 📤 Upload Data
- Loads CSV file (first 1000 rows)
- Shows data preview

### 📊 EDA
- Shows data distribution plots
- Displays correlation heatmap
- Creates time series visualizations

### 🔧 Preprocessing
- Handles missing values
- Removes outliers
- Creates new features (rolling avg, lag features)
- Normalizes data

### 🧠 Train Model
- Trains Random Forest Regressor
- Shows performance metrics (MAE, RMSE, R²)
- Displays feature importance

### 📈 Predictions
- Makes CPU load predictions
- Shows prediction errors
- Visualizes actual vs predicted

### ⚠️ Anomalies
- Detects unusual patterns using Isolation Forest
- Shows anomaly scores
- Highlights abnormal points

### ⚡ Optimization
- Runs Hill Climbing algorithm
- Performs CSP-based task scheduling
- Suggests resource optimization

---

## Expected Results

### After Running Full Pipeline:

**Model Performance:**
- MAE: ~2-3%
- RMSE: ~3-4%
- R² Score: ~0.85-0.90 ✓

**Anomalies Detected:**
- ~50 anomalies out of 1000 (5%)

**Optimization Improvement:**
- Power reduction: 10-15%
- CPU optimized to: 50-70%
- Memory optimized to: 50-75%

---

## System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|------------|
| Python | 3.7 | 3.9+ |
| RAM | 2GB | 4GB+ |
| Disk Space | 500MB | 1GB |
| Internet | Not needed | Not needed |

---

## FAQ

**Q: Can I use a different CSV file?**
A: Yes! Just ensure it has similar columns or modify the code accordingly.

**Q: How long does it take to run?**
A: ~2-3 minutes for complete pipeline (preprocess + train + predict + anomalies + optimize).

**Q: Can I modify the dataset size?**
A: Yes! In `app.py` line:
```python
global_data['df_original'] = pd.read_csv(filepath, nrows=5000)  # Change 1000 to 5000
```

**Q: How do I shut down the server?**
A: Press `Ctrl+C` in the terminal.

**Q: Where are uploaded files stored?**
A: In the `uploads/` folder.

---

## Next Steps After First Run

1. **Understand the output:** Read DOCUMENTATION.md
2. **Experiment:** Try different hyperparameters
3. **Improve:** Implement advanced techniques
4. **Deploy:** Use production WSGI server (Gunicorn)
5. **Monitor:** Add logging and metrics

---

Good luck! 🚀
