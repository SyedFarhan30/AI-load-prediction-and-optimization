"""
Flask Web Application for AI Load Prediction System
Complete backend with file upload, processing, and ML predictions

FIXES:
- preprocess route: re-loaded file by scanning upload folder; replaced with
  storing the filepath in global_data to avoid ambiguity.
- predict route: predictions is ndarray, y_actual is Series; arithmetic now
  uses np.asarray for safety.
- eda route: dtypes dict values are dtype objects (not JSON-serialisable);
  cast to str.
- data_preview route: dtypes same issue, cast to str.
- All jsonify calls: numpy scalars wrapped with float()/int() where needed.
- CORS import left in; add flask-cors to requirements if not present.
- Added /status GET endpoint so the frontend can check pipeline state.
- Added error details (trace only in DEBUG mode) for security.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import traceback
from io import BytesIO
import base64

# Import all project modules
from data_preprocessing  import DataPreprocessor, preprocess_pipeline
from load_prediction_model import LoadPredictionModel, train_load_prediction_model
from anomaly_detection   import AnomalyDetector, detect_anomalies
from optimization_module import optimize_resources, HillClimbingOptimizer, CSPScheduler
from eda_module          import EDAAnalyzer, perform_eda

# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB
app.config['UPLOAD_FOLDER']      = 'uploads'
app.config['DEBUG']              = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------
global_data = {
    'df_original':        None,
    'df_processed':       None,
    'preprocessor':       None,
    'model':              None,
    'memory_model':       None,
    'metrics':            None,
    'memory_metrics':     None,
    'anomalies':          None,
    'anomaly_detector':   None,
    'optimization_result': None,
    'feature_cols':       None,
    'eda_analyzer':       None,
    'last_file_path':     None,   # FIX: store path instead of re-scanning folder
    'df_raw_rows':        None,   # Store raw CSV rows for dashboard metrics
    'metrics_index':      0,      # Track current row index for streaming metrics
}


def _err(msg, exc=None, code=500):
    """Build a JSON error response."""
    body = {'error': msg}
    if app.config['DEBUG'] and exc:
        body['trace'] = traceback.format_exc()
    return jsonify(body), code


def _inverse_values(values, column_name):
    """Convert scaled values back to original units when preprocessor metadata exists."""
    preprocessor = global_data.get('preprocessor')
    if preprocessor is None:
        return np.asarray(values, dtype=float)
    return np.asarray(preprocessor.inverse_transform_column(values, column_name), dtype=float)


def _safe_time_labels(df, max_len):
    """Build readable anomaly/prediction locations from timestamp or row index."""
    if 'timestamp' in df.columns:
        labels = df['timestamp'].astype(str).tolist()
    else:
        labels = [f"Interval {i + 1}" for i in range(len(df))]
    return labels[:max_len]


def _build_forecast(series, horizon=20):
    """Create a short-term forecast from recent slope for dashboard display."""
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return {'horizon': horizon, 'values': [], 'trend': 'stable'}

    recent = arr[-6:] if arr.size >= 6 else arr
    diffs = np.diff(recent)
    slope = float(np.mean(diffs)) if diffs.size > 0 else 0.0

    forecast = []
    base = float(arr[-1])
    for i in range(1, horizon + 1):
        next_val = base + slope * i
        forecast.append(float(np.clip(next_val, 0.0, 100.0)))

    trend = 'stable'
    if slope > 0.2:
        trend = 'increasing'
    elif slope < -0.2:
        trend = 'decreasing'

    return {'horizon': horizon, 'values': forecast, 'trend': trend}


def _series_anomalies(series, labels, feature_name, z_thresh=2.5):
    """Detect simple feature-wise anomalies for user-friendly explanations."""
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return []

    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-9:
        return []

    z_scores = np.abs((values - mean) / std)
    indices = np.where(z_scores > z_thresh)[0]

    anomalies = []
    for idx in indices.tolist():
        value = float(values[idx])
        direction = 'surge' if value > mean else 'drop'
        anomalies.append({
            'index': int(idx),
            'location': labels[idx] if idx < len(labels) else f"Interval {idx + 1}",
            'value': value,
            'z_score': float(z_scores[idx]),
            'explanation': f"Spike detected due to sudden {feature_name} {direction}.",
        })

    return anomalies


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
@app.route('/status', methods=['GET'])
def status():
    """Return current pipeline state so the UI can enable/disable buttons."""
    return jsonify({
        'data_loaded':      global_data['df_original']   is not None,
        'preprocessed':     global_data['df_processed']  is not None,
        'model_trained':    global_data['model']         is not None,
        'anomalies_run':    global_data['anomalies']     is not None,
        'optimization_run': global_data['optimization_result'] is not None,
    }), 200


# ---------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    try:
        if 'file' not in request.files:
            return _err('No file provided', code=400)

        file = request.files['file']

        if not file.filename:
            return _err('No file selected', code=400)

        if not file.filename.lower().endswith('.csv'):
            return _err('Only CSV files are supported', code=400)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            return _err('File failed to save on server', code=500)
        
        # Read the file
        df = pd.read_csv(filepath, nrows=1000)
        global_data['df_original']    = df
        global_data['last_file_path'] = filepath  # FIX: store path
        global_data['df_raw_rows']    = df.to_dict('records')  # Store raw rows for metrics

        return jsonify({
            'message': 'File uploaded successfully',
            'shape':   list(df.shape),          # tuple → list for JSON
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records'),
            'file_path': filepath,
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Run the full preprocessing pipeline."""
    try:
        if global_data['df_original'] is None:
            return _err('No data uploaded. Call /upload first.', code=400)

        # FIX: use stored filepath, not a directory scan
        filepath = global_data['last_file_path']
        if not filepath or not os.path.exists(filepath):
            return _err('Uploaded file not found on disk.', code=400)

        df_processed, feature_cols, preprocessor = preprocess_pipeline(
            filepath, max_rows=1000
        )

        global_data['df_processed']  = df_processed
        global_data['preprocessor']  = preprocessor
        global_data['feature_cols']  = feature_cols

        original_shape = global_data['df_original'].shape
        original_ncols = original_shape[1]

        return jsonify({
            'message':          'Preprocessing completed',
            'original_shape':   list(original_shape),
            'processed_shape':  list(df_processed.shape),
            'features_total':   len(feature_cols),
            'statistics': {
                'missing_values_handled': True,
                'outliers_handled':       True,
                'features_normalized':    True,
                'new_features_created':   len(feature_cols) - original_ncols,
            },
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the load prediction model."""
    try:
        if global_data['df_processed'] is None:
            return _err('Data not preprocessed. Call /preprocess first.', code=400)

        preprocessor = global_data['preprocessor']

        # Train CPU model
        preprocessor.get_feature_columns(target_col='cpu_utilization')
        X_train, X_test, y_train, y_test = preprocessor.get_train_test_data(
            test_size=0.2, target_col='cpu_utilization'
        )

        model, metrics = train_load_prediction_model(
            X_train, X_test, y_train, y_test, model_type='random_forest'
        )

        global_data['model']   = model
        global_data['metrics'] = metrics

        # Train memory model when memory column is available
        memory_model = None
        memory_metrics = None
        if 'memory_usage' in global_data['df_processed'].columns:
            preprocessor.get_feature_columns(target_col='memory_usage')
            X_train_m, X_test_m, y_train_m, y_test_m = preprocessor.get_train_test_data(
                test_size=0.2, target_col='memory_usage'
            )
            memory_model, memory_metrics = train_load_prediction_model(
                X_train_m, X_test_m, y_train_m, y_test_m, model_type='random_forest'
            )

        global_data['memory_model'] = memory_model
        global_data['memory_metrics'] = memory_metrics

        # Default feature columns used by CPU prediction and dashboard APIs
        global_data['feature_cols'] = preprocessor.get_feature_columns(target_col='cpu_utilization')

        feat_imp = model.get_feature_importance(global_data['feature_cols'])
        
        # Format feature importance for frontend - handle various column naming conventions
        top_features = []
        if feat_imp is not None and not feat_imp.empty:
            for _, row in feat_imp.head(5).iterrows():
                # Try different possible column names
                feature_name = None
                importance_val = None
                
                if 'Feature' in row:
                    feature_name = row['Feature']
                elif 'feature' in row:
                    feature_name = row['feature']
                elif list(row.index)[0] == 0:  # If first column contains feature name
                    feature_name = list(row.values)[0]
                else:
                    feature_name = 'Feature'
                
                if 'Importance' in row:
                    importance_val = float(row['Importance'])
                elif 'importance' in row:
                    importance_val = float(row['importance'])
                elif 'value' in row:
                    importance_val = float(row['value'])
                else:
                    importance_val = 0.0
                
                top_features.append({'feature': str(feature_name), 'importance': importance_val})

        return jsonify({
            'message': 'Model trained successfully',
            'metrics': {
                'MAE':      float(metrics['MAE']),
                'RMSE':     float(metrics['RMSE']),
                'R2_Score': float(metrics['R2']),
            },
            'memory_metrics': {
                'MAE':      float(memory_metrics['MAE']),
                'RMSE':     float(memory_metrics['RMSE']),
                'R2_Score': float(memory_metrics['R2']),
            } if memory_metrics else None,
            'top_features': top_features,
        }), 200

    except Exception as exc:
        import traceback
        print(f"DEBUG train_model error: {str(exc)}")
        print(f"Traceback: {traceback.format_exc()}")
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on the processed dataset."""
    try:
        if global_data['model'] is None:
            return _err('Model not trained. Call /train_model first.', code=400)

        df = global_data['df_processed']

        # CPU predictions
        cpu_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='cpu_utilization')
        X_cpu = df[cpu_feature_cols]
        cpu_preds = global_data['model'].predict(X_cpu)
        y_cpu_actual = np.asarray(df['cpu_utilization'])

        analyzer = EDAAnalyzer(df)
        plot_image = analyzer.plot_predictions(y_cpu_actual, cpu_preds)

        cpu_preds_actual = _inverse_values(cpu_preds, 'cpu_utilization')
        y_cpu_actual_inv = _inverse_values(y_cpu_actual, 'cpu_utilization')

        cpu_mae = float(np.mean(np.abs(cpu_preds_actual - y_cpu_actual_inv)))
        cpu_mape = float(np.mean(np.abs((y_cpu_actual_inv - cpu_preds_actual) / (np.abs(y_cpu_actual_inv) + 1e-9))) * 100)
        cpu_forecast = _build_forecast(cpu_preds_actual, horizon=20)

        # Memory predictions (if memory model exists)
        memory_preds = None
        memory_stats = None
        memory_preds_actual = None
        memory_forecast = {'horizon': 20, 'values': [], 'trend': 'stable'}
        if global_data['memory_model'] is not None and 'memory_usage' in df.columns:
            memory_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='memory_usage')
            X_memory = df[memory_feature_cols]
            memory_preds = global_data['memory_model'].predict(X_memory)
            y_memory_actual = np.asarray(df['memory_usage'])

            memory_preds_actual = _inverse_values(memory_preds, 'memory_usage')
            y_memory_actual_inv = _inverse_values(y_memory_actual, 'memory_usage')

            memory_mae = float(np.mean(np.abs(memory_preds_actual - y_memory_actual_inv)))
            memory_mape = float(np.mean(np.abs((y_memory_actual_inv - memory_preds_actual) / (np.abs(y_memory_actual_inv) + 1e-9))) * 100)
            memory_forecast = _build_forecast(memory_preds_actual, horizon=20)

            memory_stats = {
                'MAE': memory_mae,
                'MAPE': memory_mape,
                'mean_prediction': float(memory_preds_actual.mean()),
                'std_prediction': float(memory_preds_actual.std()),
            }

        labels = _safe_time_labels(df, max_len=min(len(cpu_preds_actual), 50))
        cpu_display = cpu_preds_actual[:50]
        memory_display = memory_preds_actual[:50] if memory_preds_actual is not None else np.array([])

        return jsonify({
            'message':     'Predictions completed',
            'predictions': {
                'cpu': cpu_display.tolist(),
                'memory': memory_display.tolist(),
            },
            'current_predicted_usage': {
                'cpu': float(cpu_preds_actual[-1]) if len(cpu_preds_actual) else None,
                'memory': float(memory_preds_actual[-1]) if memory_preds_actual is not None and len(memory_preds_actual) else None,
            },
            'forecast': {
                'cpu': cpu_forecast,
                'memory': memory_forecast,
            },
            'labels': labels,
            'statistics': {
                'cpu': {
                    'MAE': cpu_mae,
                    'MAPE': cpu_mape,
                    'mean_prediction': float(cpu_preds_actual.mean()),
                    'std_prediction': float(cpu_preds_actual.std()),
                },
                'memory': memory_stats,
            },
            'plot': plot_image,
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies_route():
    """Detect anomalies in the processed dataset."""
    try:
        if global_data['df_processed'] is None:
            return _err('Data not preprocessed. Call /preprocess first.', code=400)

        df = global_data['df_processed']
        detector = detect_anomalies(df, method='isolation_forest', contamination=0.05)

        global_data['anomaly_detector'] = detector
        global_data['anomalies']        = detector.anomalies

        report     = detector.get_anomaly_report(df, top_n=10)
        analyzer   = EDAAnalyzer(df)
        plot_image = analyzer.plot_anomalies(detector.anomalies)
        global_data['eda_analyzer'] = analyzer

        cpu_series = _inverse_values(df['cpu_utilization'].values, 'cpu_utilization') if 'cpu_utilization' in df.columns else np.array([])
        memory_series = _inverse_values(df['memory_usage'].values, 'memory_usage') if 'memory_usage' in df.columns else np.array([])
        labels = _safe_time_labels(df, max_len=len(df))

        cpu_anomalies = _series_anomalies(cpu_series, labels, feature_name='CPU usage')
        memory_anomalies = _series_anomalies(memory_series, labels, feature_name='memory usage')

        merged = sorted(
            cpu_anomalies + memory_anomalies,
            key=lambda item: item['index']
        )

        return jsonify({
            'message': 'Anomaly detection completed',
            'report':  report,
            'plot':    plot_image,
            'summary': {
                'method':             report['method'],
                'total_anomalies':    len(merged),
                'cpu_anomalies':      len(cpu_anomalies),
                'memory_anomalies':   len(memory_anomalies),
                'anomaly_percentage': round(report['anomaly_percentage'], 2),
            },
            'anomaly_details': {
                'cpu': cpu_anomalies[:30],
                'memory': memory_anomalies[:30],
                'merged': merged[:40],
            },
            'series': {
                'labels': labels[:120],
                'cpu': cpu_series[:120].tolist(),
                'memory': memory_series[:120].tolist(),
            }
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/optimize', methods=['POST'])
def optimize():
    """Run resource optimisation (Hill Climbing + CSP scheduling)."""
    try:
        if global_data['df_processed'] is None:
            return _err('Data not preprocessed. Call /preprocess first.', code=400)
        if global_data['model'] is None:
            return _err('Model not trained. Call /train_model first.', code=400)

        df = global_data['df_processed']
        preprocessor = global_data['preprocessor']

        if preprocessor is None:
            return _err('Preprocessor is not available. Call /preprocess first.', code=400)

        cpu_feature_cols = preprocessor.get_feature_columns(target_col='cpu_utilization')
        X_cpu = df[cpu_feature_cols]
        cpu_preds = global_data['model'].predict(X_cpu)
        cpu_preds_actual = _inverse_values(cpu_preds, 'cpu_utilization')

        memory_preds = None
        memory_preds_actual = None
        if global_data['memory_model'] is not None and 'memory_usage' in df.columns:
            memory_feature_cols = preprocessor.get_feature_columns(target_col='memory_usage')
            X_memory = df[memory_feature_cols]
            memory_preds = global_data['memory_model'].predict(X_memory)
            memory_preds_actual = _inverse_values(memory_preds, 'memory_usage')

        # Use inverse-transformed values for optimisation insights shown to users.
        combined_preds = cpu_preds_actual if memory_preds_actual is None else (cpu_preds_actual + memory_preds_actual) / 2.0

        df_actual = preprocessor.inverse_transform_dataframe(
            df,
            columns=['cpu_utilization', 'memory_usage', 'power_consumption', 'network_latency', 'disk_io']
        )

        result = optimize_resources(df_actual, combined_preds, cpu_limit=80, memory_limit=85)
        global_data['optimization_result'] = result

        analyzer   = EDAAnalyzer(df_actual)
        plot_image = analyzer.plot_optimization_progress(
            result['optimizer'].optimization_history
        )

        opt = result['optimization']
        sch = result['scheduling']
        
        # Calculate optimization metrics
        current_cpu = float(np.mean(cpu_preds_actual)) if len(cpu_preds_actual) > 0 else 0.0
        current_memory = float(np.mean(memory_preds_actual)) if memory_preds_actual is not None and len(memory_preds_actual) > 0 else 0.0
        current_usage = (current_cpu + current_memory) / 2.0 if memory_preds_actual is not None else current_cpu

        optimized_cpu = float(opt.get('best_solution', {}).get('cpu_utilization', current_cpu))
        optimized_memory = float(opt.get('best_solution', {}).get('memory_usage', current_memory)) if memory_preds_actual is not None else None
        optimized_usage = (optimized_cpu + optimized_memory) / 2.0 if optimized_memory is not None else optimized_cpu
        potential_savings = ((current_usage - optimized_usage) / (current_usage + 1e-9)) * 100

        cpu_limit_reco = int(np.clip(round(max(75, min(85, optimized_cpu + 5))), 75, 85))
        memory_limit_reco = int(np.clip(round(max(80, min(90, (optimized_memory if optimized_memory is not None else current_memory) + 5))), 80, 90))

        return jsonify({
            'message':         'Optimisation completed',
            'optimization': {
                'current_usage':       current_usage,
                'current_cpu_usage':   current_cpu,
                'current_memory_usage': current_memory if memory_preds_actual is not None else None,
                'optimized_usage':     optimized_usage,
                'optimized_cpu_usage': optimized_cpu,
                'optimized_memory_usage': optimized_memory,
                'potential_savings':   potential_savings,
                'cpu_limit':           cpu_limit_reco,
                'memory_limit':        memory_limit_reco,
                'recommendations':     opt.get('recommendations', []),
            },
            'scheduling': {
                'strategy':            'Optimal Task Scheduling',
                'tasks_count':         sch.get('total_tasks', 0),
                'scheduled_count':     sch.get('scheduled_count', 0),
                'success_rate':        sch.get('success_rate', 0),
                'suggestion':          'Prioritize high-CPU tasks in lower-load intervals and cap concurrent memory-heavy jobs.' if sch.get('scheduled_count', 0) > 0 else 'No scheduling actions required for current load.',
            },
            'plot': plot_image,
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/eda', methods=['POST'])
def eda():
    """Run EDA and return analysis + visualisations."""
    try:
        if global_data['df_original'] is None:
            return _err('No data uploaded. Please upload a CSV file first.', code=400)

        df       = global_data['df_original']
        analyzer = EDAAnalyzer(df)

        analysis = analyzer.analyze_dataset()
        insights = analyzer.get_insights()

        dist_plot = analyzer.plot_distributions()
        corr_plot = analyzer.plot_correlations()
        ts_plot   = analyzer.plot_time_series()

        return jsonify({
            'message': 'EDA completed',
            'analysis': {
                'shape':   list(analysis['shape']),
                # FIX: limit to first 5 columns; dtypes cast to str
                'columns': analysis['columns'][:5],
                'dtypes':  {k: str(v) for k, v in analysis['dtypes'].items()},
            },
            'insights': {k: v for k, v in insights.items() if v is not None},
            'plots': {
                'distributions': dist_plot,
                'correlations':  corr_plot,
                'time_series':   ts_plot,
            },
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/data_preview', methods=['GET'])
def data_preview():
    """Return a preview of the raw uploaded data."""
    try:
        if global_data['df_original'] is None:
            return _err('No data loaded.', code=400)

        df = global_data['df_original']

        return jsonify({
            'columns': list(df.columns),
            'data':    df.head(20).to_dict('records'),
            'shape':   list(df.shape),
            # FIX: dtype objects → str
            'dtypes':  df.dtypes.astype(str).to_dict(),
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/reset', methods=['POST'])
def reset():
    """Reset all pipeline state and clean uploads."""
    global global_data
    global_data = {
        'df_original':         None,
        'df_processed':        None,
        'preprocessor':        None,
        'model':               None,
        'memory_model':        None,
        'metrics':             None,
        'memory_metrics':      None,
        'anomalies':           None,
        'anomaly_detector':    None,
        'optimization_result': None,
        'feature_cols':        None,
        'eda_analyzer':        None,
        'last_file_path':      None,
        'df_raw_rows':         None,
        'metrics_index':       0,
    }

    upload_dir = app.config['UPLOAD_FOLDER']
    for fname in os.listdir(upload_dir):
        fpath = os.path.join(upload_dir, fname)
        try:
            os.remove(fpath)
        except OSError:
            pass

    return jsonify({'message': 'System reset successfully'}), 200


# ---------------------------------------------------------------------------
# NEW API ENDPOINTS FOR DASHBOARD
# ---------------------------------------------------------------------------

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return current system metrics for the dashboard from uploaded CSV data."""
    try:
        # If we have raw CSV rows, use them for metrics
        if global_data['df_raw_rows'] and len(global_data['df_raw_rows']) > 0:
            # Get the next row cyclically
            idx = global_data['metrics_index'] % len(global_data['df_raw_rows'])
            row = global_data['df_raw_rows'][idx]
            global_data['metrics_index'] += 1
            
            # Extract metrics from row
            cpu_usage = float(row.get('cpu_utilization', 50))
            memory_usage = float(row.get('memory_usage', 60))
            network_traffic = float(row.get('network_latency', 50))
            
            return jsonify({
                'timestamp': pd.Timestamp.now().isoformat(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_traffic': network_traffic,
                'status': 'operational',
                'source': 'csv_data'
            }), 200
        
        # Fallback to dataset statistics if available
        elif global_data['df_processed'] is not None:
            df = global_data['df_processed']
            cpu_mean = float(df['cpu_utilization'].mean()) if 'cpu_utilization' in df.columns else 50.0
            memory_mean = float(df['memory_usage'].mean()) if 'memory_usage' in df.columns else 60.0
            network_mean = float(df['network_latency'].mean()) if 'network_latency' in df.columns else 50.0
            
            return jsonify({
                'timestamp': pd.Timestamp.now().isoformat(),
                'cpu_usage': cpu_mean,
                'memory_usage': memory_mean,
                'network_traffic': network_mean,
                'status': 'operational',
                'source': 'dataset_stats'
            }), 200
        
        # Default fallback
        else:
            return jsonify({
                'timestamp': pd.Timestamp.now().isoformat(),
                'cpu_usage': 50.0,
                'memory_usage': 60.0,
                'network_traffic': 50.0,
                'status': 'waiting_for_data',
                'source': 'default'
            }), 200
    except Exception as exc:
        return _err(str(exc), exc)


@app.route('/api/predictions_summary', methods=['GET'])
def api_predictions_summary():
    """Return summary of last predictions."""
    try:
        if global_data['model'] is None or global_data['df_processed'] is None:
            return jsonify({
                'predictions': {'cpu': [], 'memory': []},
                'latest_prediction': None,
                'average_confidence': 0,
                'anomalies_detected': 0
            }), 200
        
        df = global_data['df_processed']
        cpu_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='cpu_utilization')
        X_cpu = df[cpu_feature_cols]
        cpu_preds = global_data['model'].predict(X_cpu)

        memory_preds = np.array([])
        if global_data['memory_model'] is not None and 'memory_usage' in df.columns:
            memory_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='memory_usage')
            X_memory = df[memory_feature_cols]
            memory_preds = global_data['memory_model'].predict(X_memory)

        combined_latest = float(cpu_preds[-1]) if len(cpu_preds) > 0 else None
        combined_avg = float(np.mean(cpu_preds)) if len(cpu_preds) > 0 else 0.0
        if len(memory_preds) > 0:
            combined_latest = float((cpu_preds[-1] + memory_preds[-1]) / 2.0)
            combined_avg = float((np.mean(cpu_preds) + np.mean(memory_preds)) / 2.0)
        
        return jsonify({
            'predictions': {
                'cpu': cpu_preds[:10].tolist(),
                'memory': memory_preds[:10].tolist() if len(memory_preds) > 0 else [],
            },
            'latest_prediction': combined_latest,
            'average_confidence': combined_avg,
            'anomalies_detected': int(len(global_data['anomalies'])) if global_data['anomalies'] is not None else 0
        }), 200
    except Exception as exc:
        return _err(str(exc), exc)


@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """Return current alerts."""
    try:
        alerts = []
        
        if global_data['df_processed'] is not None and global_data['model'] is not None:
            df = global_data['df_processed']
            cpu_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='cpu_utilization')
            X_cpu = df[cpu_feature_cols]
            cpu_preds = global_data['model'].predict(X_cpu)

            avg_cpu = float(np.mean(cpu_preds)) if len(cpu_preds) > 0 else 0.0
            avg_memory = 0.0
            if global_data['memory_model'] is not None and 'memory_usage' in df.columns:
                memory_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='memory_usage')
                X_memory = df[memory_feature_cols]
                memory_preds = global_data['memory_model'].predict(X_memory)
                avg_memory = float(np.mean(memory_preds)) if len(memory_preds) > 0 else 0.0

            avg_pred = (avg_cpu + avg_memory) / 2.0 if avg_memory > 0 else avg_cpu
            if avg_pred > 80:
                    alerts.append({
                        'id': 1,
                        'severity': 'critical',
                        'message': f'High predicted load (CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%): {avg_pred:.1f}%',
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
            elif avg_pred > 60:
                    alerts.append({
                        'id': 2,
                        'severity': 'warning',
                        'message': f'Elevated predicted load (CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%): {avg_pred:.1f}%',
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
        
        return jsonify({'alerts': alerts}), 200
    except Exception as exc:
        return _err(str(exc), exc)


@app.route('/api/history', methods=['GET'])
def api_history():
    """Return prediction history."""
    try:
        if global_data['df_processed'] is None or global_data['model'] is None:
            return jsonify({'history': []}), 200
        
        df = global_data['df_processed']
        cpu_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='cpu_utilization')
        X_cpu = df[cpu_feature_cols]
        cpu_preds = global_data['model'].predict(X_cpu)
        y_cpu_actual = df['cpu_utilization'].values

        memory_preds = None
        y_memory_actual = None
        if global_data['memory_model'] is not None and 'memory_usage' in df.columns:
            memory_feature_cols = global_data['preprocessor'].get_feature_columns(target_col='memory_usage')
            X_memory = df[memory_feature_cols]
            memory_preds = global_data['memory_model'].predict(X_memory)
            y_memory_actual = df['memory_usage'].values
        
        history = []
        for i in range(min(20, len(cpu_preds))):
            combined_pred = float(cpu_preds[i])
            if memory_preds is not None and i < len(memory_preds):
                combined_pred = float((cpu_preds[i] + memory_preds[i]) / 2.0)

            load_level = 'Low' if combined_pred < 40 else 'Medium' if combined_pred < 60 else 'High' if combined_pred < 80 else 'Critical'
            is_anomaly = global_data['anomalies'][i] if global_data['anomalies'] is not None and i < len(global_data['anomalies']) else False
            
            history.append({
                'time': f"{pd.Timestamp.now() - pd.Timedelta(minutes=20-i)}".split('.')[0],
                'predicted': combined_pred,
                'actual': float(y_cpu_actual[i]),
                'cpu_predicted': float(cpu_preds[i]),
                'cpu_actual': float(y_cpu_actual[i]),
                'memory_predicted': float(memory_preds[i]) if memory_preds is not None and i < len(memory_preds) else None,
                'memory_actual': float(y_memory_actual[i]) if y_memory_actual is not None and i < len(y_memory_actual) else None,
                'load_level': load_level,
                'anomaly': bool(is_anomaly),
                'confidence': float(np.abs(cpu_preds[i] - y_cpu_actual[i]))
            })
        
        return jsonify({'history': history}), 200
    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)