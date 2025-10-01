"""Time-based feature engineering plugin.

Creates cyclic time features from datetime columns:
- Hour of day (sine/cosine encoding)
- Day of year (sine/cosine encoding)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from core.plugin_system import FeatureEngineeringPlugin

logger = logging.getLogger(__name__)


class TimeFeaturesPlugin(FeatureEngineeringPlugin):
    """Plugin for creating time-based features."""
    
    def __init__(self):
        super().__init__(
            name="time_features",
            version="1.0.0",
            description="Creates cyclic time features (hour, day of year)"
        )
        
    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add cyclic time features to DataFrame.
        
        Args:
            df: Input DataFrame with TIME column
            config: Configuration containing time feature settings
            
        Returns:
            DataFrame with added time features
        """
        df_copy = df.copy()
        
        time_config = config.get('time_features', {})
        if not time_config.get('enabled', True):
            logger.info("Time features disabled in configuration")
            return df_copy
            
        time_col = config.get('time_column', 'TIME')
        
        if time_col not in df_copy.columns:
            logger.warning(f"Time column '{time_col}' not found in DataFrame")
            return df_copy
            
        logger.info("Adding time features: hour_sin/cos, doy_sin/cos.")
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
            logger.warning("TIME column is not datetime, converting.")
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            
        # Extract temporal components
        df_copy['hour'] = df_copy[time_col].dt.hour
        df_copy['dayofyear'] = df_copy[time_col].dt.dayofyear
        
        # Create cyclic features
        if time_config.get('cyclic_hour', True):
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
            
        if time_config.get('cyclic_day_of_year', True):
            df_copy['doy_sin'] = np.sin(2 * np.pi * df_copy['dayofyear'] / 365)
            df_copy['doy_cos'] = np.cos(2 * np.pi * df_copy['dayofyear'] / 365)
            
        # Remove temporary columns
        df_copy = df_copy.drop(columns=['hour', 'dayofyear'])
        
        logger.info("Finished adding time features.")
        return df_copy
        
    def get_feature_names(self, input_features: List[str], config: Dict[str, Any]) -> List[str]:
        """Get names of time features that will be created."""
        feature_names = []
        time_config = config.get('time_features', {})
        
        if time_config.get('cyclic_hour', True):
            feature_names.extend(['hour_sin', 'hour_cos'])
            
        if time_config.get('cyclic_day_of_year', True):
            feature_names.extend(['doy_sin', 'doy_cos'])
            
        return feature_names
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate time features configuration."""
        time_config = config.get('time_features', {})
        
        # Check if at least one time feature is enabled
        if (not time_config.get('cyclic_hour', True) and 
            not time_config.get('cyclic_day_of_year', True)):
            logger.warning("No time features enabled")
            return False
            
        return True