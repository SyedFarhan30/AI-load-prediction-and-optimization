"""
Data Preprocessing Module for AI Load Prediction System
Handles data loading, cleaning, and feature engineering

FIXES:
- handle_missing_values: fillna(df.mean()) only works on numeric cols; now filters properly
- normalize_features: 'status' and 'timestamp' should be excluded from scaling
- get_feature_columns: excludes non-numeric cols more robustly
- get_train_test_data: target_col removed from feature list to avoid data leakage
- engineer_features: gracefully skips missing columns
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing tasks."""

    def __init__(self, file_path=None, max_rows=1000):
        self.file_path = file_path
        self.max_rows = max_rows
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.feature_cols = []

    # ------------------------------------------------------------------
    def load_data(self):
        """Load data from CSV file (only first max_rows rows)."""
        self.df = pd.read_csv(self.file_path, nrows=self.max_rows)
        print(f"✓ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    # ------------------------------------------------------------------
    def analyze_data(self):
        """Analyze dataset and provide basic statistics."""
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),   # FIX: dtype objects aren't JSON-serialisable
            'missing_values': self.df.isnull().sum().to_dict(),
            'statistics': self.df.describe().to_dict(),
            'duplicates': int(self.df.duplicated().sum()),
        }
        return analysis

    # ------------------------------------------------------------------
    def handle_missing_values(self, method='mean'):
        """
        Handle missing values in dataset.

        FIX: original code called df.mean() on the whole frame which raises a
        TypeError when non-numeric columns are present.  Now we only fill
        numeric columns, and leave non-numeric ones as-is.
        """
        df_copy = self.df.copy()

        if df_copy.isnull().sum().sum() == 0:
            print("✓ No missing values found")
            self.df_processed = df_copy
            return df_copy

        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        if method == 'mean':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            print("✓ Missing values filled using mean")
        elif method == 'median':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
            print("✓ Missing values filled using median")
        elif method == 'drop':
            df_copy.dropna(inplace=True)
            print("✓ Rows with missing values dropped")
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'mean', 'median', or 'drop'.")

        self.df_processed = df_copy
        return df_copy

    # ------------------------------------------------------------------
    def handle_outliers(self, method='iqr', threshold=1.5):
        """Handle outliers using IQR or Z-score method."""
        df_copy = self.df_processed.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
            print("✓ Outliers handled using IQR method")

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[numeric_cols]))
            for i, col in enumerate(numeric_cols):
                mask = z_scores[:, i] > threshold
                df_copy.loc[mask, col] = df_copy[col].median()
            print("✓ Outliers handled using Z-score method")

        self.df_processed = df_copy
        return df_copy

    # ------------------------------------------------------------------
    def engineer_features(self):
        """
        Create new features from existing ones.

        FIX: ratio features referenced hard-coded column names that may not
        exist.  Now we check existence before creating each ratio.
        FIX: final fillna uses numeric mean only.
        """
        df_copy = self.df_processed.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        exclude = {'status'}

        # Rolling averages & std (3-row window)
        for col in numeric_cols:
            if col not in exclude:
                df_copy[f'{col}_rolling_avg'] = (
                    df_copy[col].rolling(window=3, min_periods=1).mean()
                )
                df_copy[f'{col}_rolling_std'] = (
                    df_copy[col].rolling(window=3, min_periods=1).std().fillna(0)
                )

        # Lag features (1-step lag)
        for col in numeric_cols:
            if col not in exclude:
                df_copy[f'{col}_lag1'] = df_copy[col].shift(1).fillna(df_copy[col])

        # Ratio features – only created when both columns exist
        _cols = set(df_copy.columns)
        if {'cpu_utilization', 'memory_usage'}.issubset(_cols):
            df_copy['cpu_memory_ratio'] = (
                df_copy['cpu_utilization'] / (df_copy['memory_usage'] + 1e-9)
            )
        if {'disk_io', 'network_latency'}.issubset(_cols):
            df_copy['io_latency_ratio'] = (
                df_copy['disk_io'] / (df_copy['network_latency'] + 1e-9)
            )
        if {'thread_count', 'process_count'}.issubset(_cols):
            df_copy['thread_process_ratio'] = (
                df_copy['thread_count'] / (df_copy['process_count'] + 1e-9)
            )

        # FIX: fill NaNs only on numeric columns
        num_cols_new = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[num_cols_new] = df_copy[num_cols_new].fillna(df_copy[num_cols_new].mean())

        print(f"✓ Features engineered: {df_copy.shape[1]} total features")
        self.df_processed = df_copy
        return df_copy

    # ------------------------------------------------------------------
    def normalize_features(self, method='standard'):
        """
        Normalize/Scale features.

        FIX: excluded 'status' and 'timestamp' from scaling in the original
        code only by name – but the scaler was applied to ALL numeric cols
        including 'status'.  Now we explicitly build the list of cols to scale.
        """
        df_copy = self.df_processed.copy()

        # Columns to scale: numeric, excluding identifiers / binary labels
        skip_cols = {'status', 'timestamp'}
        scale_cols = [
            c for c in df_copy.select_dtypes(include=[np.number]).columns
            if c not in skip_cols
        ]

        if method == 'standard':
            df_copy[scale_cols] = self.scaler.fit_transform(df_copy[scale_cols])
            print("✓ Features normalized using StandardScaler")
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_copy[scale_cols] = scaler.fit_transform(df_copy[scale_cols])
            print("✓ Features normalized using MinMaxScaler")
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'standard' or 'minmax'.")

        self.df_processed = df_copy
        return df_copy

    # ------------------------------------------------------------------
    def get_feature_columns(self, target_col='cpu_utilization'):
        """
        Get list of feature columns for modelling.

        FIX: original excluded only 'status' and 'timestamp', leaving the
        target column inside the feature list → data leakage.  Now we also
        exclude the target.
        """
        exclude = {'status', 'timestamp', target_col}
        feature_cols = [
            col for col in self.df_processed.columns
            if col not in exclude
            and self.df_processed[col].dtype in [np.float64, np.float32,
                                                  np.int64, np.int32]
        ]
        self.feature_cols = feature_cols
        return feature_cols

    # ------------------------------------------------------------------
    def get_processed_data(self):
        return self.df_processed

    # ------------------------------------------------------------------
    def get_train_test_data(self, test_size=0.2, target_col='cpu_utilization'):
        """
        Split data into train and test sets.

        FIX: feature list is rebuilt here to ensure target is excluded.
        """
        # Rebuild feature list without the target to avoid leakage
        if not self.feature_cols:
            self.get_feature_columns(target_col=target_col)

        # Also remove target from feature cols just in case
        safe_features = [c for c in self.feature_cols if c != target_col]

        X = self.df_processed[safe_features]
        y = self.df_processed[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------
def preprocess_pipeline(file_path, max_rows=1000):
    """
    Complete preprocessing pipeline.

    Returns:
        (df_processed, feature_cols, preprocessor)
    """
    preprocessor = DataPreprocessor(file_path, max_rows)
    preprocessor.load_data()
    preprocessor.handle_missing_values(method='mean')
    preprocessor.handle_outliers(method='iqr', threshold=1.5)
    preprocessor.engineer_features()
    preprocessor.normalize_features(method='standard')
    feature_cols = preprocessor.get_feature_columns()
    return preprocessor.get_processed_data(), feature_cols, preprocessor