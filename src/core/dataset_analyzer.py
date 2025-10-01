"""Dataset introspection and automatic configuration generation.

This module analyzes datasets and automatically generates appropriate configurations
for different data sources, making the pipeline truly dataset-agnostic.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes datasets and generates appropriate configurations."""
    
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.data_shape = None
        self.time_resolution = None
        self.date_range = None
        
    def analyze_dataset(self, data_path: str, sample_size: int = 10000) -> Dict[str, Any]:
        """Analyze dataset and return configuration recommendations.
        
        Args:
            data_path: Path to the dataset
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Dictionary with dataset analysis and config recommendations
        """
        logger.info(f"Analyzing dataset: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        # Load sample of data
        try:
            df_sample = pd.read_csv(data_path, nrows=sample_size)
        except Exception as e:
            raise ValueError(f"Failed to read dataset: {e}")
            
        # Basic dataset info
        self.data_shape = df_sample.shape
        logger.info(f"Dataset shape (sample): {self.data_shape}")
        
        # Analyze column types
        column_analysis = self._analyze_columns(df_sample)
        
        # Detect time column and resolution
        time_analysis = self._analyze_time_structure(df_sample)
        
        # Suggest target column
        target_analysis = self._suggest_target_column(df_sample)
        
        # Suggest feature engineering parameters
        feature_engineering_config = self._suggest_feature_engineering(
            df_sample, time_analysis
        )
        
        # Generate periods based on date range
        periods_config = self._suggest_periods(time_analysis)
        
        # Combine all analyses
        analysis_result = {
            'dataset_info': {
                'path': data_path,
                'shape': list(self.data_shape),  # Convert tuple to list for YAML compatibility
                'analysis_date': datetime.now().isoformat()
            },
            'columns': column_analysis,
            'time_analysis': time_analysis,
            'target_suggestions': target_analysis,
            'recommended_config': {
                'data': {
                    'data_path': data_path,
                    'time_column': time_analysis.get('recommended_time_column'),
                    'target_column': target_analysis.get('recommended_target'),
                    'feature_columns': column_analysis.get('recommended_features', []),
                    'exclude_cols': [time_analysis.get('recommended_time_column')] if time_analysis.get('recommended_time_column') else [],
                },
                'periods': periods_config,
                'feature_engineering': feature_engineering_config
            }
        }
        
        return analysis_result
        
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and characteristics."""
        column_info = {}
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique()),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None
                })
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)
                
            # Check if column might be datetime
            if self._is_datetime_column(df[col]):
                self.datetime_columns.append(col)
                col_info['is_datetime'] = True
                
            column_info[col] = col_info
            
        # Recommend feature columns (numeric, non-target, non-time)
        # First detect target column
        likely_targets = [col for col in self.numeric_columns if self._is_likely_target(col)]
        
        recommended_features = [
            col for col in self.numeric_columns 
            if col not in self.datetime_columns and col not in likely_targets
        ]
        
        return {
            'column_details': column_info,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'recommended_features': recommended_features
        }
        
    def _analyze_time_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time structure of the dataset."""
        time_analysis = {}
        
        # Find the most likely time column
        time_column = self._detect_time_column(df)
        
        if time_column:
            time_analysis['recommended_time_column'] = time_column
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                try:
                    time_series = pd.to_datetime(df[time_column])
                except:
                    logger.warning(f"Could not convert {time_column} to datetime")
                    return time_analysis
            else:
                time_series = df[time_column]
                
            # Analyze time resolution
            time_resolution = self._detect_time_resolution(time_series)
            time_analysis['resolution'] = time_resolution
            
            # Date range
            time_analysis['date_range'] = {
                'start': str(time_series.min()),
                'end': str(time_series.max()),
                'duration_days': (time_series.max() - time_series.min()).days
            }
            
            # Seasonal patterns
            time_analysis['seasonal_info'] = self._analyze_seasonal_patterns(time_series)
            
        return time_analysis
        
    def _suggest_target_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Suggest potential target columns."""
        target_candidates = []
        
        for col in self.numeric_columns:
            if self._is_likely_target(col):
                target_candidates.append({
                    'column': col,
                    'confidence': self._calculate_target_confidence(col, df[col])
                })
                
        # Sort by confidence
        target_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'candidates': target_candidates,
            'recommended_target': target_candidates[0]['column'] if target_candidates else None
        }
        
    def _suggest_feature_engineering(self, df: pd.DataFrame, time_analysis: Dict) -> Dict[str, Any]:
        """Suggest feature engineering parameters based on dataset characteristics."""
        
        # Get time resolution for lag/rolling suggestions
        resolution_minutes = self._get_resolution_minutes(time_analysis.get('resolution'))
        
        # Suggest lag periods (multiple time scales)
        lag_suggestions = self._suggest_lag_periods(resolution_minutes)
        
        # Suggest rolling windows (multiple time scales)
        rolling_suggestions = self._suggest_rolling_windows(resolution_minutes)
        
        # Generate configuration
        fe_config = {
            'time_features': {
                'enabled': True,
                'cyclic_hour': True,
                'cyclic_day_of_year': True
            },
            'lag_features': {
                'enabled': True,
                'auto_lags': lag_suggestions,
                'default_lags': self._create_default_lags(lag_suggestions)
            },
            'rolling_features': {
                'enabled': True,
                'auto_windows': rolling_suggestions,
                'default_windows': self._create_default_windows(rolling_suggestions),
                'statistics': ['mean', 'std']
            }
        }
        
        return fe_config
        
    def _suggest_periods(self, time_analysis: Dict) -> List[Dict[str, str]]:
        """Suggest analysis periods based on data date range."""
        if not time_analysis.get('date_range'):
            return []
            
        start_date = pd.to_datetime(time_analysis['date_range']['start'])
        end_date = pd.to_datetime(time_analysis['date_range']['end'])
        duration_days = time_analysis['date_range']['duration_days']
        
        periods = []
        
        if duration_days > 365:
            # Year-long data or more - suggest seasonal periods
            periods = [
                {
                    'name': 'Winter',
                    'start_date': '12-01',
                    'end_date': '02-28',
                    'description': 'Winter period (Dec-Feb)'
                },
                {
                    'name': 'Spring',
                    'start_date': '03-01',
                    'end_date': '05-31',
                    'description': 'Spring period (Mar-May)'  
                },
                {
                    'name': 'Summer',
                    'start_date': '06-01',
                    'end_date': '08-31',
                    'description': 'Summer period (Jun-Aug)'
                },
                {
                    'name': 'Autumn',
                    'start_date': '09-01',
                    'end_date': '11-30',
                    'description': 'Autumn period (Sep-Nov)'
                }
            ]
        elif duration_days > 90:
            # Quarterly periods
            quarters = self._generate_quarterly_periods(start_date, end_date)
            periods.extend(quarters)
        else:
            # Monthly periods for shorter datasets
            months = self._generate_monthly_periods(start_date, end_date)
            periods.extend(months)
            
        return periods
        
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the most likely time column."""
        time_indicators = ['time', 'date', 'timestamp', 'datetime', 'created_at', 'updated_at']
        
        # First check for obvious time column names
        for col in df.columns:
            if any(indicator in col.lower() for indicator in time_indicators):
                if self._is_datetime_column(df[col]):
                    return col
                    
        # Check all columns for datetime-like content
        for col in df.columns:
            if self._is_datetime_column(df[col]):
                return col
                
        return None
        
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a series contains datetime data."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
            
        # Try parsing a sample
        try:
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False
                
            pd.to_datetime(sample)
            return True
        except:
            return False
            
    def _is_likely_target(self, column_name: str) -> bool:
        """Check if column name suggests it's a target variable."""
        target_indicators = [
            'target', 'label', 'y', 'output', 'prediction', 'class',
            'n2o', 'emission', 'concentration', 'level', 'amount'
        ]
        
        return any(indicator in column_name.lower() for indicator in target_indicators)
        
    def _calculate_target_confidence(self, column_name: str, series: pd.Series) -> float:
        """Calculate confidence that this column is the target."""
        confidence = 0.0
        
        # Name-based confidence
        if self._is_likely_target(column_name):
            confidence += 0.6
            
        # Statistical characteristics
        if series.dtype in ['float64', 'float32']:
            confidence += 0.2
            
        # Variability (targets often have good variability)
        if not series.isnull().all():
            cv = series.std() / abs(series.mean()) if series.mean() != 0 else 0
            if 0.1 < cv < 2.0:  # Good coefficient of variation
                confidence += 0.2
                
        return min(confidence, 1.0)
        
    def _detect_time_resolution(self, time_series: pd.Series) -> str:
        """Detect the time resolution of the data."""
        if len(time_series) < 2:
            return "unknown"
            
        # Calculate time differences
        time_diffs = time_series.sort_values().diff().dropna()
        
        if len(time_diffs) == 0:
            return "unknown"
            
        # Find the most common time difference
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) == 0:
            return "unknown"
            
        seconds = mode_diff.iloc[0].total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)}S"
        elif seconds < 3600:
            return f"{int(seconds/60)}T"
        elif seconds < 86400:
            return f"{int(seconds/3600)}H"
        else:
            return f"{int(seconds/86400)}D"
            
    def _get_resolution_minutes(self, resolution: str) -> int:
        """Convert resolution string to minutes."""
        if not resolution or resolution == "unknown":
            return 15  # Default assumption
            
        if resolution.endswith('S'):
            return max(1, int(resolution[:-1]) // 60)
        elif resolution.endswith('T'):
            return int(resolution[:-1])
        elif resolution.endswith('H'):
            return int(resolution[:-1]) * 60
        elif resolution.endswith('D'):
            return int(resolution[:-1]) * 1440
        else:
            return 15
            
    def _suggest_lag_periods(self, resolution_minutes: int) -> Dict[str, List[int]]:
        """Suggest lag periods based on time resolution."""
        # Define meaningful time scales in minutes
        time_scales = {
            'immediate': resolution_minutes,
            'short': resolution_minutes * 4,
            'medium': resolution_minutes * 16,
            'long': resolution_minutes * 64
        }
        
        # Convert to lag steps
        lags = []
        for scale_minutes in time_scales.values():
            lag_steps = max(1, scale_minutes // resolution_minutes)
            if lag_steps not in lags and lag_steps <= 100:  # Reasonable limit
                lags.append(lag_steps)
                
        return {'suggested_lags': sorted(lags)[:6]}  # Limit to 6 lags
        
    def _suggest_rolling_windows(self, resolution_minutes: int) -> Dict[str, List[int]]:
        """Suggest rolling window sizes based on time resolution."""
        # Define meaningful time windows in minutes  
        window_scales = {
            'short': 60,      # 1 hour
            'medium': 360,    # 6 hours
            'long': 1440,     # 24 hours
            'very_long': 10080 # 1 week
        }
        
        # Convert to window steps
        windows = []
        for scale_minutes in window_scales.values():
            window_steps = max(1, scale_minutes // resolution_minutes)
            if window_steps not in windows and window_steps <= 1000:  # Reasonable limit
                windows.append(window_steps)
                
        return {'suggested_windows': sorted(windows)[:4]}  # Limit to 4 windows
        
    def _create_default_lags(self, lag_suggestions: Dict) -> Dict[str, List[int]]:
        """Create default lag configuration for all numeric features."""
        lags = lag_suggestions.get('suggested_lags', [1, 2])
        return {'AUTO_FEATURES': lags}  # Special key for auto-detected features
        
    def _create_default_windows(self, rolling_suggestions: Dict) -> Dict[str, List[int]]:
        """Create default rolling window configuration for all numeric features.""" 
        windows = rolling_suggestions.get('suggested_windows', [6, 24])
        return {'AUTO_FEATURES': windows}  # Special key for auto-detected features
        
    def _analyze_seasonal_patterns(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns in the time series."""
        # This is a placeholder for more sophisticated seasonal analysis
        return {
            'has_daily_pattern': True,  # Most sensor data has daily patterns
            'has_weekly_pattern': False,  # Depends on the process
            'has_yearly_pattern': (time_series.max() - time_series.min()).days > 300
        }
        
    def _generate_quarterly_periods(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict]:
        """Generate quarterly periods for the data range."""
        periods = []
        year = start_date.year
        
        quarters = [
            ('Q1', '01-01', '03-31'),
            ('Q2', '04-01', '06-30'), 
            ('Q3', '07-01', '09-30'),
            ('Q4', '10-01', '12-31')
        ]
        
        for q_name, start_mmdd, end_mmdd in quarters:
            periods.append({
                'name': f'{year}_{q_name}',
                'start_date': start_mmdd,
                'end_date': end_mmdd,
                'description': f'{q_name} {year} ({start_mmdd} to {end_mmdd})'
            })
            
        return periods
        
    def _generate_monthly_periods(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict]:
        """Generate monthly periods for the data range."""
        periods = []
        current = start_date.replace(day=1)
        
        while current <= end_date:
            month_end = (current + pd.DateOffset(months=1) - pd.DateOffset(days=1))
            periods.append({
                'name': current.strftime('%Y_%m'),
                'start_date': current.strftime('%m-01'),
                'end_date': month_end.strftime('%m-%d'),
                'description': current.strftime('%B %Y')
            })
            current += pd.DateOffset(months=1)
            
        return periods
        
    def generate_config_file(self, analysis_result: Dict, output_path: str) -> None:
        """Generate a YAML configuration file based on analysis."""
        
        # Create base configuration template
        config = {
            'general': {
                'project_name': f"ML_Pipeline_{datetime.now().strftime('%Y%m%d')}",
                'random_seed': 42,
                'verbose': True,
                'save_results': True,
                'save_figures': True
            },
            
            'data': analysis_result['recommended_config']['data'],
            
            'periods': analysis_result['recommended_config']['periods'],
            
            'models': {
                'available_models': ['KNN', 'RandomForest', 'XGBoost', 'AdaBoost', 'DecisionTree', 'DNN'],
                'default_model': 'RandomForest',  # Good general choice
                'cv_folds': 5,
                'cv_scoring': 'r2',
                'cv_n_jobs': -1,
                'cv_verbose': 2
            },
            
            'feature_engineering': analysis_result['recommended_config']['feature_engineering'],
            
            'augmentation': {
                'enabled': False,
                'n_augment': 0,
                'noise_level': 0.0
            },
            
            'resampling': {
                'default_rule': None,
                'model_overrides': {}
            },
            
            'output': {
                'results_dir': 'results',
                'model_output_dir': 'results/model_outputs',
                'figures_dir': 'results/figures', 
                'logs_dir': 'results/logs',
                'predictions_dir': 'results/predictions',
                'model_format': 'joblib',
                'results_format': 'csv'
            },
            
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': True,
                'console_logging': True,
                'log_file': 'results/logs/pipeline.log'
            },
            
            'hyperparameters': {
                'RandomForest': {
                    'n_estimators': [100, 500],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 10]
                },
                'XGBoost': {
                    'n_estimators': [100, 500],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 6]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 10]
                }
            },
            
            'plugins': {
                'feature_engineering': {
                    'enabled_plugins': ['time_features', 'lag_features', 'rolling_features'],
                    'plugin_search_paths': ['src/plugins/feature_engineering']
                }
            },
            
            # Add dataset analysis metadata
            'dataset_analysis': {
                'analysis_date': analysis_result['dataset_info']['analysis_date'],
                'original_shape': analysis_result['dataset_info']['shape'],
                'detected_columns': analysis_result['columns'],
                'time_analysis': analysis_result['time_analysis']
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save configuration
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            
        logger.info(f"Generated configuration file: {output_path}")


def analyze_and_configure(data_path: str, output_config_path: str = None) -> str:
    """Analyze dataset and generate configuration file.
    
    Args:
        data_path: Path to the dataset to analyze
        output_config_path: Path for output config file (auto-generated if None)
        
    Returns:
        Path to the generated configuration file
    """
    analyzer = DatasetAnalyzer()
    
    # Perform analysis
    analysis_result = analyzer.analyze_dataset(data_path)
    
    # Generate output path if not provided
    if output_config_path is None:
        dataset_name = Path(data_path).stem
        output_config_path = f"config/auto_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    # Generate configuration file
    analyzer.generate_config_file(analysis_result, output_config_path)
    
    return output_config_path


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_analyzer.py <data_path> [output_config_path]")
        sys.exit(1)
        
    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        config_path = analyze_and_configure(data_path, output_path)
        print(f"✅ Configuration generated: {config_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)