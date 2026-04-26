"""
Exploratory Data Analysis (EDA) Module
Creates visualizations and insights from the dataset

FIXES:
- analyze_dataset: printed 'Missing values: duplicates' twice (wrong key)
- get_insights: hard-coded column names crash if columns missing; now safe
- plot_predictions: y_pred may be ndarray, not Series → .values guard added
- All plot methods: column-existence checks added
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # FIX: use non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)


class EDAAnalyzer:
    """Performs exploratory data analysis with visualizations."""

    def __init__(self, df):
        self.df = df
        self.insights = {}

    # ------------------------------------------------------------------
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis."""
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),   # FIX: JSON-serialisable
            'missing': self.df.isnull().sum().to_dict(),
            'duplicates': int(self.df.duplicated().sum()),
            'statistics': self.df.describe().to_dict(),
        }

        print(f"✓ Dataset Analysis:")
        print(f"  Shape: {analysis['shape']}")
        # FIX: original printed analysis['duplicates'] for BOTH missing & duplicates
        print(f"  Missing values: {sum(analysis['missing'].values())}")
        print(f"  Duplicates: {analysis['duplicates']}")

        self.insights['analysis'] = analysis
        return analysis

    # ------------------------------------------------------------------
    def plot_distributions(self, columns=None):
        """Create distribution plots for numeric columns."""
        if columns is None:
            columns = ['cpu_utilization', 'memory_usage',
                       'network_latency', 'power_consumption']

        # FIX: only keep columns that actually exist
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return None

        n = len(columns)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = np.array(axes).ravel()

        for idx, col in enumerate(columns):
            axes[idx].hist(self.df[col].dropna(), bins=30,
                           color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def plot_correlations(self):
        """Create correlation heatmap."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax,
                    cbar_kws={'label': 'Correlation'},
                    annot_kws={'size': 7})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def plot_time_series(self, columns=None):
        """Create time series plots."""
        if columns is None:
            columns = ['cpu_utilization', 'memory_usage', 'power_consumption']

        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return None

        fig, axes = plt.subplots(len(columns), 1, figsize=(14, 4 * len(columns)))
        if len(columns) == 1:
            axes = [axes]

        for idx, col in enumerate(columns):
            axes[idx].plot(self.df[col], linewidth=1.5, color='steelblue')
            axes[idx].set_xlabel('Time Index')
            axes[idx].set_ylabel(col)
            axes[idx].set_title(f'{col} Over Time')
            axes[idx].grid(True, alpha=0.3)

            rolling_avg = self.df[col].rolling(window=20).mean()
            axes[idx].plot(rolling_avg, linewidth=2, color='red',
                           label='20-point MA', alpha=0.7)
            axes[idx].legend()

        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def plot_anomalies(self, anomalies, columns=None):
        """Plot normal vs anomalous points."""
        if columns is None:
            columns = ['cpu_utilization', 'memory_usage', 'network_latency']

        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return None

        fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))
        if len(columns) == 1:
            axes = [axes]

        # FIX: anomalies may be a pandas Series or ndarray; normalise to ndarray
        anomalies_arr = np.asarray(anomalies, dtype=bool)

        for idx, col in enumerate(columns):
            axes[idx].scatter(
                np.where(~anomalies_arr)[0],
                self.df[col].values[~anomalies_arr],
                c='blue', alpha=0.4, label='Normal', s=15
            )
            axes[idx].scatter(
                np.where(anomalies_arr)[0],
                self.df[col].values[anomalies_arr],
                c='red', alpha=0.8, label='Anomaly', s=50, marker='X'
            )
            axes[idx].set_xlabel('Index')
            axes[idx].set_ylabel(col)
            axes[idx].set_title(f'{col} – Anomaly Detection')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def plot_predictions(self, y_true, y_pred):
        """
        Plot actual vs predicted values.

        FIX: y_pred is usually an ndarray; calling .values on it crashes.
             Now we cast both to numpy arrays uniformly.
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        n = min(200, len(y_true_arr))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(y_true_arr[:n], label='Actual', linewidth=2, color='green')
        ax1.plot(y_pred_arr[:n], label='Predicted', linewidth=2,
                 color='orange', alpha=0.7)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('CPU Utilization')
        ax1.set_title(f'Actual vs Predicted CPU Load (First {n} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_true_arr, y_pred_arr, alpha=0.5, s=15)
        min_val = min(y_true_arr.min(), y_pred_arr.min())
        max_val = max(y_true_arr.max(), y_pred_arr.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--',
                 lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Prediction Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def plot_optimization_progress(self, history):
        """Plot optimization progress."""
        if not history:
            return None

        iterations     = [h['iteration']     for h in history]
        current_costs  = [h['current_cost']  for h in history]
        neighbor_costs = [h['neighbor_cost'] for h in history]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(iterations, current_costs,  label='Current Cost',  linewidth=2, color='blue')
        ax.plot(iterations, neighbor_costs, label='Neighbor Cost', linewidth=2,
                color='orange', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Hill Climbing Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------
    def get_insights(self):
        """
        Generate data insights.

        FIX: original hard-coded column names; now uses .get() with defaults
        so missing columns don't raise KeyError.
        """
        def _safe_stat(col, func):
            if col in self.df.columns:
                return float(func(self.df[col].dropna()))
            return None

        insights = {
            'high_cpu':     _safe_stat('cpu_utilization',   max),
            'avg_memory':   _safe_stat('memory_usage',      np.mean),
            'high_latency': _safe_stat('network_latency',   max),
            'avg_power':    _safe_stat('power_consumption', np.mean),
        }

        if 'uptime' in self.df.columns:
            insights['uptime_range'] = (
                float(self.df['uptime'].min()),
                float(self.df['uptime'].max()),
            )

        return insights


# -----------------------------------------------------------------------
def fig_to_base64(fig):
    """Convert matplotlib figure to base64-encoded PNG string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


# -----------------------------------------------------------------------
def perform_eda(df, max_samples=None):
    """Perform complete EDA on dataset."""
    if max_samples:
        df = df.head(max_samples)

    analyzer = EDAAnalyzer(df)
    print("✓ Starting Exploratory Data Analysis...")
    analyzer.analyze_dataset()
    return analyzer