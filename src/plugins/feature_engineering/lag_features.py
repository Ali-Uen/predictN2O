"""Lag features engineering plugin.

Creates lagged versions of specified columns for time series modeling.
Supports configurable lag periods per column.
"""

import pandas as pd
import logging
from typing import Dict, List, Any
from core.plugin_system import FeatureEngineeringPlugin

logger = logging.getLogger(__name__)


class LagFeaturesPlugin(FeatureEngineeringPlugin):
    """Plugin for creating lag features."""
    
    def __init__(self):
        super().__init__(
            name="lag_features",
            version="1.0.0",
            description="Creates lagged versions of specified columns"
        )
        
    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add lag features to DataFrame.
        
        Args:
            df: Input DataFrame
            config: Configuration containing lag feature settings
            
        Returns:
            DataFrame with added lag features
        """
        df_copy = df.copy()
        
        lag_config = config.get('lag_features', {})
        if not lag_config.get('enabled', True):
            logger.info("Lag features disabled in configuration")
            return df_copy
            
        # Get lag configuration - can be period/model specific
        lags_dict = self._get_lags_dict(config)
        
        if not lags_dict:
            logger.info("No lag features configured")
            return df_copy
            
        logger.info(f"Adding lag features: {lags_dict}")
        
        for col, lags in lags_dict.items():
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' not found for lag features")
                continue
                
            for lag in lags:
                new_col = f'{col}_lag{lag}'
                df_copy[new_col] = df_copy[col].shift(lag)
                logger.debug(f"Added lag feature: {new_col}")
                
        logger.info("Finished adding lag features.")
        return df_copy
        
    def _get_lags_dict(self, config: Dict[str, Any]) -> Dict[str, List[int]]:
        """Get lag configuration from config.
        
        Priority:
        1. Model and period specific lags (if available)
        2. Default lags from config
        3. Auto-detected lags for all numeric features
        4. Empty dict
        
        Args:
            config: Full configuration
            
        Returns:
            Dictionary mapping column names to lag periods
        """
        # Try to get model/period specific lags (for backward compatibility)
        model_name = config.get('models', {}).get('default_model')
        period_name = config.get('current_period_name')  # Set during processing
        
        if model_name and period_name:
            # Look for legacy model_period_lags structure
            model_lags = config.get('model_period_lags', {})
            if model_name in model_lags and period_name in model_lags[model_name]:
                return model_lags[model_name][period_name]
                
        # Get default lags from config
        lag_config = config.get('lag_features', {})
        default_lags = lag_config.get('default_lags', {})
        
        # If we have AUTO_FEATURES key, apply to all numeric features
        if 'AUTO_FEATURES' in default_lags:
            auto_lag_values = default_lags['AUTO_FEATURES']
            feature_columns = config.get('data', {}).get('feature_columns', [])
            
            # Apply auto lags to all feature columns
            expanded_lags = {}
            for col in feature_columns:
                expanded_lags[col] = auto_lag_values
                
            # Merge with any explicitly defined lags
            for col, lags in default_lags.items():
                if col != 'AUTO_FEATURES':
                    expanded_lags[col] = lags
                    
            return expanded_lags
            
        return default_lags
        
    def get_feature_names(self, input_features: List[str], config: Dict[str, Any]) -> List[str]:
        """Get names of lag features that will be created."""
        lags_dict = self._get_lags_dict(config)
        feature_names = []
        
        for col, lags in lags_dict.items():
            for lag in lags:
                feature_names.append(f'{col}_lag{lag}')
                
        return feature_names
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate lag features configuration."""
        lags_dict = self._get_lags_dict(config)
        
        # Check for valid lag periods
        for col, lags in lags_dict.items():
            if not isinstance(lags, list):
                logger.error(f"Lags for column '{col}' must be a list")
                return False
                
            for lag in lags:
                if not isinstance(lag, int) or lag <= 0:
                    logger.error(f"Invalid lag period for column '{col}': {lag}")
                    return False
                    
        return True