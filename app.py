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
    'metrics':            None,
    'anomalies':          None,
    'anomaly_detector':   None,
    'optimization_result': None,
    'feature_cols':       None,
    'eda_analyzer':       None,
    'last_file_path':     None,   # FIX: store path instead of re-scanning folder
}


def _err(msg, exc=None, code=500):
    """Build a JSON error response."""
    body = {'error': msg}
    if app.config['DEBUG'] and exc:
        body['trace'] = traceback.format_exc()
    return jsonify(body), code


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
        file.save(filepath)

        df = pd.read_csv(filepath, nrows=1000)
        global_data['df_original']    = df
        global_data['last_file_path'] = filepath  # FIX: store path

        return jsonify({
            'message': 'File uploaded successfully',
            'shape':   list(df.shape),          # tuple → list for JSON
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records'),
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

        original_ncols = global_data['df_original'].shape[1]

        return jsonify({
            'message':          'Preprocessing completed',
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

        X_train, X_test, y_train, y_test = global_data['preprocessor'].get_train_test_data(
            test_size=0.2, target_col='cpu_utilization'
        )

        model, metrics = train_load_prediction_model(
            X_train, X_test, y_train, y_test, model_type='random_forest'
        )

        global_data['model']   = model
        global_data['metrics'] = metrics

        feat_imp = model.get_feature_importance(global_data['feature_cols'])

        return jsonify({
            'message': 'Model trained successfully',
            'metrics': {
                'MAE':      float(metrics['MAE']),
                'RMSE':     float(metrics['RMSE']),
                'R2_Score': float(metrics['R2']),
            },
            'top_features': (
                feat_imp.head(5).to_dict('records') if feat_imp is not None else []
            ),
        }), 200

    except Exception as exc:
        return _err(str(exc), exc)


# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on the processed dataset."""
    try:
        if global_data['model'] is None:
            return _err('Model not trained. Call /train_model first.', code=400)

        df   = global_data['df_processed']
        X    = df[global_data['feature_cols']]
        preds = global_data['model'].predict(X)          # ndarray

        y_actual = np.asarray(df['cpu_utilization'])     # FIX: ensure ndarray

        analyzer   = EDAAnalyzer(df)
        plot_image = analyzer.plot_predictions(y_actual, preds)

        mae  = float(np.mean(np.abs(preds - y_actual)))
        # FIX: avoid division by zero
        mape = float(np.mean(np.abs((y_actual - preds) / (np.abs(y_actual) + 1e-9))) * 100)

        return jsonify({
            'message':     'Predictions completed',
            'predictions': preds[:50].tolist(),
            'statistics': {
                'MAE':             mae,
                'MAPE':            mape,
                'mean_prediction': float(preds.mean()),
                'std_prediction':  float(preds.std()),
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

        df       = global_data['df_processed']
        detector = detect_anomalies(df, method='isolation_forest', contamination=0.05)

        global_data['anomaly_detector'] = detector
        global_data['anomalies']        = detector.anomalies

        report     = detector.get_anomaly_report(df, top_n=10)
        analyzer   = EDAAnalyzer(df)
        plot_image = analyzer.plot_anomalies(detector.anomalies)
        global_data['eda_analyzer'] = analyzer

        return jsonify({
            'message': 'Anomaly detection completed',
            'report':  report,
            'plot':    plot_image,
            'summary': {
                'method':             report['method'],
                'total_anomalies':    report['anomalies_count'],
                'anomaly_percentage': round(report['anomaly_percentage'], 2),
            },
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

        df    = global_data['df_processed']
        X     = df[global_data['feature_cols']]
        preds = global_data['model'].predict(X)

        result = optimize_resources(df, preds, cpu_limit=80, memory_limit=85)
        global_data['optimization_result'] = result

        analyzer   = EDAAnalyzer(df)
        plot_image = analyzer.plot_optimization_progress(
            result['optimizer'].optimization_history
        )

        opt = result['optimization']
        sch = result['scheduling']

        return jsonify({
            'message':         'Optimisation completed',
            'best_solution':   opt['best_solution'],
            'best_cost':       float(opt['best_cost']),
            'improvement':     opt['improvement'],
            'recommendations': opt['recommendations'],
            'scheduling': {
                'scheduled_count': sch['scheduled_count'],
                'total_tasks':     sch['total_tasks'],
                'success_rate':    round(sch['success_rate'], 2),
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
        if global_data['df_processed'] is None:
            return _err('Data not preprocessed. Call /preprocess first.', code=400)

        df       = global_data['df_processed']
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
        'metrics':             None,
        'anomalies':           None,
        'anomaly_detector':    None,
        'optimization_result': None,
        'feature_cols':        None,
        'eda_analyzer':        None,
        'last_file_path':      None,
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
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)