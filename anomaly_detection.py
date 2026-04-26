"""
Anomaly Detection Module
Detects unusual system performance patterns

FIXES:
- detect_statistical_anomalies: zscore() returns an ndarray, not a DataFrame;
  any(axis=1) must be called on a DataFrame or 2-D array – fixed.
- get_anomaly_details: anomaly_scores may be 1-D ndarray indexed with a
  boolean mask – use np.asarray + boolean indexing consistently.
- get_anomaly_report: row['row_index'] may be float; cast to int safely.
- Removed implicit assumption that 'cpu_utilization' / 'memory_usage' /
  'network_latency' always exist in the anomaly details DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detects anomalies in system performance metrics."""

    def __init__(self, method='statistical', contamination=0.05):
        self.method = method
        self.contamination = contamination
        self.anomaly_scores = None   # 1-D ndarray, length == len(df)
        self.anomalies = None        # 1-D bool ndarray
        self.if_model = None

    # ------------------------------------------------------------------
    def detect_statistical_anomalies(self, df, columns=None, z_threshold=3):
        """
        Detect anomalies using Z-score.

        FIX: scipy.stats.zscore returns an ndarray; we need .any(axis=1) on
        a 2-D array, not on a DataFrame index.
        """
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)

        df_numeric = df[columns].copy().fillna(df[columns].mean())
        z_scores = np.abs(stats.zscore(df_numeric.values, nan_policy='omit'))

        # FIX: z_scores is (n_samples, n_features); reduce across features
        anomalies = (z_scores > z_threshold).any(axis=1)

        self.anomalies = anomalies
        self.anomaly_scores = z_scores.max(axis=1)

        print("✓ Statistical anomaly detection completed")
        print(f"  Anomalies found: {anomalies.sum()} "
              f"({anomalies.sum() / len(df) * 100:.2f}%)")
        return anomalies

    # ------------------------------------------------------------------
    def detect_isolation_forest_anomalies(self, df, columns=None):
        """Detect anomalies using Isolation Forest."""
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)

        df_numeric = df[columns].copy().fillna(df[columns].mean())

        self.if_model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds = self.if_model.fit_predict(df_numeric)
        anomalies = (preds == -1)

        self.anomalies = anomalies
        self.anomaly_scores = -self.if_model.score_samples(df_numeric)

        print("✓ Isolation Forest anomaly detection completed")
        print(f"  Anomalies found: {anomalies.sum()} "
              f"({anomalies.sum() / len(df) * 100:.2f}%)")
        return anomalies

    # ------------------------------------------------------------------
    def detect(self, df, columns=None):
        """Detect anomalies using the configured method."""
        if self.method == 'statistical':
            return self.detect_statistical_anomalies(df, columns)
        elif self.method == 'isolation_forest':
            return self.detect_isolation_forest_anomalies(df, columns)
        else:
            raise ValueError(f"Unknown method '{self.method}'. "
                             "Choose 'statistical' or 'isolation_forest'.")

    # ------------------------------------------------------------------
    def get_anomaly_details(self, df, top_n=10):
        """
        Get details of the most significant anomalies.

        FIX: boolean-index both the DataFrame and the scores array using the
        same mask so lengths stay aligned.
        """
        if self.anomalies is None:
            raise ValueError("Call detect() before get_anomaly_details().")

        mask = np.asarray(self.anomalies, dtype=bool)
        scores = np.asarray(self.anomaly_scores)

        anomaly_df = df[mask].copy()
        anomaly_df['anomaly_score'] = scores[mask]
        anomaly_df['row_index'] = np.where(mask)[0]

        return anomaly_df.sort_values('anomaly_score', ascending=False).head(top_n)

    # ------------------------------------------------------------------
    def get_anomaly_report(self, df, top_n=10):
        """Generate an anomaly detection report."""
        anomaly_details = self.get_anomaly_details(df, top_n)

        report = {
            'method':             self.method,
            'total_records':      len(df),
            'anomalies_count':    int(np.asarray(self.anomalies).sum()),
            'anomaly_percentage': float(
                np.asarray(self.anomalies).sum() / len(df) * 100
            ),
            'top_anomalies': [],
        }

        # FIX: use .get() with defaults so missing columns don't crash
        detail_cols = {
            'cpu_utilization':  'cpu_utilization',
            'memory_usage':     'memory_usage',
            'network_latency':  'network_latency',
        }

        for _, row in anomaly_details.iterrows():
            entry = {
                'index':         int(row['row_index']),
                'anomaly_score': float(row['anomaly_score']),
            }
            for key, col in detail_cols.items():
                entry[key] = float(row[col]) if col in row.index else None
            report['top_anomalies'].append(entry)

        return report


# -----------------------------------------------------------------------
def detect_anomalies(df, method='statistical', contamination=0.05):
    """
    Convenience wrapper: detect anomalies in df.

    Returns:
        AnomalyDetector with results populated.
    """
    detector = AnomalyDetector(method=method, contamination=contamination)
    detector.detect(df)
    return detector


# -----------------------------------------------------------------------
def get_anomaly_visualization_data(df, anomalies, columns=None):
    """Prepare data for anomaly visualisation."""
    if columns is None:
        columns = ['cpu_utilization', 'memory_usage', 'network_latency']

    anomalies_arr = np.asarray(anomalies, dtype=bool)
    data = {'normal': [], 'anomaly': []}

    for col in columns:
        if col in df.columns:
            data['normal'].append({
                'column': col,
                'values': df[col].values[~anomalies_arr][:100].tolist(),
            })
            data['anomaly'].append({
                'column': col,
                'values': df[col].values[anomalies_arr][:100].tolist(),
            })

    return data