"""Rolling window features engineering plugin.

Creates rolling statistics (mean, std) over specified time windows.
Supports configurable window sizes per column.
"""

import pandas as pd
import logging
from typing import Dict, List, Any
from core.plugin_system import FeatureEngineeringPlugin

logger = logging.getLogger(__name__)


class RollingFeaturesPlugin(FeatureEngineeringPlugin):
    """Plugin for creating rolling window features."""
    
    def __init__(self):
        super().__init__(
            name="rolling_features",
            version="1.0.0",
            description="Creates rolling statistics (mean, std) over time windows"
        )
        
    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add rolling features to DataFrame.
        
        Args:
            df: Input DataFrame
            config: Configuration containing rolling feature settings
            
        Returns:
            DataFrame with added rolling features
        """
        df_copy = df.copy()
        
        rolling_config = config.get('rolling_features', {})
        if not rolling_config.get('enabled', True):
            logger.info("Rolling features disabled in configuration")
            return df_copy
            
        # Get rolling configuration - can be period/model specific
        rolling_dict = self._get_rolling_dict(config)
        statistics = rolling_config.get('statistics', ['mean', 'std'])
        
        if not rolling_dict:
            logger.info("No rolling features configured")
            return df_copy
            
        logger.info(f"Adding rolling features: {rolling_dict}")
        
        for col, windows in rolling_dict.items():
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' not found for rolling features")
                continue
                
            for window in windows:
                # Create rolling statistics
                rolling_obj = df_copy[col].rolling(window, min_periods=1)
                
                if 'mean' in statistics:
                    df_copy[f"{col}_roll{window}"] = rolling_obj.mean()
                    
                if 'std' in statistics:
                    df_copy[f"{col}_std{window}"] = rolling_obj.std()
                    
                # Additional statistics if configured
                if 'min' in statistics:
                    df_copy[f"{col}_min{window}"] = rolling_obj.min()
                    
                if 'max' in statistics:
                    df_copy[f"{col}_max{window}"] = rolling_obj.max()
                    
                if 'median' in statistics:
                    df_copy[f"{col}_med{window}"] = rolling_obj.median()
                    
                logger.debug(f"Added rolling features for {col} with window {window}")
                
        logger.info("Finished adding rolling features.")
        return df_copy
        
    def _get_rolling_dict(self, config: Dict[str, Any]) -> Dict[str, List[int]]:
        """Get rolling window configuration from config.
        
        Priority:
        1. Model and period specific windows (if available)
        2. Default windows from config
        3. Auto-detected windows for all numeric features
        4. Empty dict
        
        Args:
            config: Full configuration
            
        Returns:
            Dictionary mapping column names to window sizes
        """
        # Try to get model/period specific windows (for backward compatibility)
        model_name = config.get('models', {}).get('default_model')
        period_name = config.get('current_period_name')  # Set during processing
        
        if model_name and period_name:
            # Look for legacy model_period_rolling structure
            model_rolling = config.get('model_period_rolling', {})
            if model_name in model_rolling and period_name in model_rolling[model_name]:
                return model_rolling[model_name][period_name]
                
        # Get default windows from config
        rolling_config = config.get('rolling_features', {})
        default_windows = rolling_config.get('default_windows', {})
        
        # If we have AUTO_FEATURES key, apply to all numeric features
        if 'AUTO_FEATURES' in default_windows:
            auto_window_values = default_windows['AUTO_FEATURES']
            feature_columns = config.get('data', {}).get('feature_columns', [])
            
            # Apply auto windows to all feature columns
            expanded_windows = {}
            for col in feature_columns:
                expanded_windows[col] = auto_window_values
                
            # Merge with any explicitly defined windows
            for col, windows in default_windows.items():
                if col != 'AUTO_FEATURES':
                    expanded_windows[col] = windows
                    
            return expanded_windows
            
        return default_windows
        
    def get_feature_names(self, input_features: List[str], config: Dict[str, Any]) -> List[str]:
        """Get names of rolling features that will be created."""
        rolling_dict = self._get_rolling_dict(config)
        rolling_config = config.get('rolling_features', {})
        statistics = rolling_config.get('statistics', ['mean', 'std'])
        
        feature_names = []
        
        for col, windows in rolling_dict.items():
            for window in windows:
                if 'mean' in statistics:
                    feature_names.append(f"{col}_roll{window}")
                if 'std' in statistics:
                    feature_names.append(f"{col}_std{window}")
                if 'min' in statistics:
                    feature_names.append(f"{col}_min{window}")
                if 'max' in statistics:
                    feature_names.append(f"{col}_max{window}")
                if 'median' in statistics:
                    feature_names.append(f"{col}_med{window}")
                    
        return feature_names
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate rolling features configuration."""
        rolling_dict = self._get_rolling_dict(config)
        rolling_config = config.get('rolling_features', {})
        statistics = rolling_config.get('statistics', ['mean', 'std'])
        
        # Check for valid window sizes
        for col, windows in rolling_dict.items():
            if not isinstance(windows, list):
                logger.error(f"Windows for column '{col}' must be a list")
                return False
                
            for window in windows:
                if not isinstance(window, int) or window <= 0:
                    logger.error(f"Invalid window size for column '{col}': {window}")
                    return False
                    
        # Check for valid statistics
        valid_stats = ['mean', 'std', 'min', 'max', 'median']
        for stat in statistics:
            if stat not in valid_stats:
                logger.error(f"Invalid statistic: {stat}. Valid options: {valid_stats}")
                return False
                
        return True