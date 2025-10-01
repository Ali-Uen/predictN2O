"""
Streamlit Web Interface for N2O Emission Prediction Pipeline

This web application provides an interactive interface for:
1. Dataset loading and exploration
2. Data preprocessing configuration  
3. Model selection and training
4. Results visualization and analysis

Usage:
    streamlit run web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import tempfile
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import core modules from src/
try:
    from core.config_manager import ConfigManager
    from core.plugin_system import PluginRegistry
    from data_loader import load_and_prepare_data, remove_outliers_iqr, remove_outliers_mahalanobis
    from models import train_model, estimator_dict
    from evaluation import evaluate_regression
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported core modules from src/")
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")
    st.stop()

from src.data_loader import load_and_prepare_data
from src.core.dataset_analyzer import DatasetAnalyzer
from src.models import train_model, train_dnn_model
from src.evaluation import evaluate_regression
from src.augmentation import add_noise
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Configure page
st.set_page_config(
    page_title="N2O Emission Prediction Pipeline",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 1
    st.session_state.data = None
    st.session_state.analysis = None
    st.session_state.processed_data = None
    st.session_state.model_results = None
    st.session_state.training_cancelled = False
    st.session_state.training_in_progress = False
    st.session_state.feature_selection_completed = False
    
    # Initialize saved settings
    st.session_state.saved_settings = {
        'preprocessing': {
            'remove_nan': True,
            'remove_negative': True,
            'remove_outliers': True,
            'outlier_method': 'IQR (Interquartile Range)'
        },
        'model_config': {
            'model': 'RandomForest',
            'train_split': 0.8,
            'scaler_type': 'StandardScaler',
            'feature_engineering': {
                'enabled': True,
                'time_features': True,
                'lag_features': True,
                'lag_periods': [1, 4, 16],
                'rolling_features': True,
                'rolling_windows': [4, 24, 96]
            },
            'augmentation': {
                'enabled': True,
                'n_augment': 3,
                'noise_level': 0.05
            },
            'hyperparameters': {}
        }
    }

def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4682B4;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4682B4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32cd32;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fffacd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffd700;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_core_modules():
    """Initialize ConfigManager and PluginRegistry."""
    try:
        # Initialize ConfigManager
        config_manager = ConfigManager()
        
        # Initialize PluginRegistry
        plugin_registry = PluginRegistry()
        plugin_registry.load_plugins_from_directory("src/plugins/feature_engineering", "feature_engineering")
        
        logger.info("Core modules initialized successfully")
        return config_manager, plugin_registry
    except Exception as e:
        logger.error(f"Failed to initialize core modules: {e}")
        return None, None

def apply_core_preprocessing(df: pd.DataFrame, options: Dict) -> pd.DataFrame:
    """Apply preprocessing using core data_loader functions."""
    try:
        processed_df = df.copy()
        original_shape = processed_df.shape[0]
        
        # Remove NaN values
        if options.get('remove_nan', True):
            processed_df = processed_df.dropna()
            logger.info(f"Removed NaN values: {original_shape - processed_df.shape[0]} rows")
        
        # Remove negative values (assuming numeric columns except TIME)
        if options.get('remove_negative', True):
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove TIME column if present
            if 'TIME' in numeric_cols:
                numeric_cols.remove('TIME')
            
            if numeric_cols:
                mask = (processed_df[numeric_cols] >= 0).all(axis=1)
                processed_df = processed_df[mask].reset_index(drop=True)
                logger.info(f"Removed negative values: {original_shape - processed_df.shape[0]} rows total")
        
        # Remove outliers using core functions
        if options.get('remove_outliers', True):
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'TIME' in numeric_cols:
                numeric_cols.remove('TIME')
            
            if numeric_cols:
                outlier_method = options.get('outlier_method', 'IQR (Interquartile Range)')
                
                if outlier_method == "IQR (Interquartile Range)":
                    processed_df = remove_outliers_iqr(processed_df, numeric_cols)
                elif outlier_method == "Mahalanobis Distance":
                    processed_df = remove_outliers_mahalanobis(processed_df, numeric_cols)
                elif outlier_method == "Both":
                    processed_df = remove_outliers_iqr(processed_df, numeric_cols)
                    processed_df = remove_outliers_mahalanobis(processed_df, numeric_cols)
        
        logger.info(f"Final preprocessing result: {processed_df.shape[0]} rows remaining from {original_shape}")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in apply_core_preprocessing: {e}")
        raise

def apply_feature_engineering_plugins(df: pd.DataFrame, fe_config: Dict) -> pd.DataFrame:
    """Apply feature engineering using plugin system."""
    try:
        processed_df = df.copy()
        plugin_registry = st.session_state.plugin_registry
        
        if not fe_config.get('enabled', False):
            return processed_df
        
        # Time features
        if fe_config.get('time_features', False):
            time_plugin = plugin_registry.get_plugin('time_features')
            if time_plugin:
                config = {
                    'time_column': 'TIME',
                    'time_features': {
                        'enabled': True,
                        'cyclic_hour': True,
                        'cyclic_day_of_year': True
                    }
                }
                processed_df = time_plugin.transform(processed_df, config)
                logger.info("Applied time features plugin")
        
        # Lag features  
        if fe_config.get('lag_features', False):
            lag_plugin = plugin_registry.get_plugin('lag_features')
            if lag_plugin:
                config = {
                    'lag_features': {
                        'enabled': True,
                        'periods': fe_config.get('lag_periods', [1, 4, 16]),
                        'target_column': 'N2O_mg_per_l'  # This should be configurable
                    }
                }
                processed_df = lag_plugin.transform(processed_df, config)
                logger.info(f"Applied lag features plugin with periods {config['lag_features']['periods']}")
        
        # Rolling features
        if fe_config.get('rolling_features', False):
            rolling_plugin = plugin_registry.get_plugin('rolling_features')
            if rolling_plugin:
                config = {
                    'rolling_features': {
                        'enabled': True,
                        'windows': fe_config.get('rolling_windows', [4, 24, 96]),
                        'target_column': 'N2O_mg_per_l'  # This should be configurable
                    }
                }
                processed_df = rolling_plugin.transform(processed_df, config)
                logger.info(f"Applied rolling features plugin with windows {config['rolling_features']['windows']}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in apply_feature_engineering_plugins: {e}")
        # Return original dataframe if plugin fails
        return df

def main():
    """Main application function."""
    load_css()
    
    # Initialize core modules
    config_manager, plugin_registry = initialize_core_modules()
    if config_manager is None or plugin_registry is None:
        st.error("Failed to initialize core modules. Please check the logs.")
        return
    
    # Store in session state for access in other functions
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = config_manager
        st.session_state.plugin_registry = plugin_registry
    
    # Title and description
    st.markdown('<h1 class="main-header">🌱 N2O Emission Prediction Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
    Interactive web interface for machine learning-based nitrous oxide emission prediction
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📋 Pipeline Stages")
        
        stages = [
            "1️⃣ Dataset Loading & Analysis",
            "2️⃣ Feature Selection",
            "3️⃣ Model Configuration", 
            "4️⃣ Training & Results"
        ]
        
        selected_stage = st.radio(
            "Select Stage:",
            stages,
            index=st.session_state.stage - 1
        )
        
        st.session_state.stage = stages.index(selected_stage) + 1
        
        # Progress indicator
        progress = (st.session_state.stage - 1) / (len(stages) - 1)
        st.progress(progress)
        
        st.markdown("---")
        st.markdown("### 📊 Current Status")
        
        status_items = [
            ("Dataset Loaded", st.session_state.data is not None, "✅" if st.session_state.data is not None else "⏳"),
            ("Analysis Complete", st.session_state.analysis is not None, "✅" if st.session_state.analysis is not None else "⏳"),
            ("Features Selected", st.session_state.feature_selection_completed, "✅" if st.session_state.feature_selection_completed else "⏳"),
            ("Model Trained", st.session_state.model_results is not None, "✅" if st.session_state.model_results is not None else "⏳")
        ]
        
        for item, status, icon in status_items:
            st.markdown(f"{icon} {item}")
    
    # Main content based on selected stage
    if st.session_state.stage == 1:
        stage_1_dataset_loading()
    elif st.session_state.stage == 2:
        stage_2_feature_selection()
    elif st.session_state.stage == 3:
        stage_3_model_configuration()
    elif st.session_state.stage == 4:
        stage_4_training_results()

def stage_1_dataset_loading():
    """Stage 1: Dataset Loading and Analysis."""
    st.markdown('<div class="section-header">1️⃣ Dataset Loading & Analysis</div>', unsafe_allow_html=True)
    
    # File upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your emission dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file containing time series data with emissions measurements"
        )
    
    with col2:
        use_example = st.button("🔬 Use Example Dataset", help="Load the Altenrhein WWTP dataset")
    
    # Load data
    if uploaded_file is not None:
        try:
            with st.spinner("Loading dataset..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load data with basic CSV reading first
                df = pd.read_csv(tmp_path)
                st.session_state.data = df
                st.session_state.data_path = tmp_path
                st.session_state.raw_data = df.copy()  # Keep original for comparison
                
                # Clean up
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
            
    elif use_example:
        try:
            with st.spinner("Loading example dataset..."):
                example_path = "data/AltenrheinWWTP.csv"
                if os.path.exists(example_path):
                    df = pd.read_csv(example_path)
                    st.session_state.data = df
                    st.session_state.data_path = example_path
                    st.session_state.raw_data = df.copy()  # Keep original for comparison
                else:
                    st.error("Example dataset not found. Please upload your own dataset.")
                    return
        except Exception as e:
            st.error(f"Error loading example dataset: {str(e)}")
            return
    
    # Display dataset analysis if data is loaded
    if st.session_state.data is not None:
        display_dataset_analysis()

def display_dataset_analysis():
    """Display comprehensive dataset analysis."""
    df = st.session_state.data
    
    st.success("✅ Dataset loaded successfully!")
    
    # Basic dataset characteristics
    st.markdown('<div class="section-header">📊 Dataset Characteristics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Column analysis
    st.markdown('<div class="section-header">🔍 Column Analysis</div>', unsafe_allow_html=True)
    
    # Analyze columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col not in numeric_cols:
            # Try to detect datetime
            try:
                pd.to_datetime(df[col].head(100))
                datetime_cols.append(col)
            except:
                categorical_cols.append(col)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**📈 Numeric Columns ({len(numeric_cols)}):**")
        for col in numeric_cols:
            st.write(f"• {col}")
    
    with col2:
        st.markdown(f"**📅 DateTime Columns ({len(datetime_cols)}):**")
        for col in datetime_cols:
            st.write(f"• {col}")
    
    with col3:
        st.markdown(f"**📝 Categorical Columns ({len(categorical_cols)}):**")
        for col in categorical_cols:
            st.write(f"• {col}")
    
    # Column selection for analysis
    st.markdown('<div class="section-header">⚙️ Column Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_column = st.selectbox(
            "Select Time Column:",
            datetime_cols + df.columns.tolist(),
            help="Choose the column containing timestamps"
        )
        
        target_column = st.selectbox(
            "Select Target Column (Emission Variable):",
            numeric_cols,
            help="Choose the emission variable you want to predict"
        )
    
    with col2:
        feature_columns = st.multiselect(
            "Select Feature Columns:",
            [col for col in numeric_cols if col != target_column],
            default=[col for col in numeric_cols if col != target_column],
            help="Choose the input features for prediction"
        )
    
    # Store selections
    st.session_state.time_column = time_column
    st.session_state.target_column = target_column
    st.session_state.feature_columns = feature_columns
    
    # Feature correlation analysis
    if target_column and feature_columns:
        st.markdown('<div class="section-header">🔗 Feature Correlation Analysis</div>', unsafe_allow_html=True)
        
        # Calculate correlations
        correlation_data = []
        for feature in feature_columns:
            if feature in df.columns and target_column in df.columns:
                corr = df[feature].corr(df[target_column])
                correlation_data.append({"Feature": feature, "Correlation": corr})
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            corr_df = corr_df.sort_values("Correlation", key=abs, ascending=False)
            
            # Correlation bar chart
            fig = px.bar(
                corr_df, 
                x="Feature", 
                y="Correlation", 
                title=f"Feature Correlation with {target_column}",
                color="Correlation",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation table
            st.dataframe(corr_df, use_container_width=True)
    
    # Target variable overview
    if target_column:
        st.markdown('<div class="section-header">🎯 Target Variable Analysis</div>', unsafe_allow_html=True)
        
        target_data = df[target_column].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Statistics
            st.markdown("**📊 Statistics:**")
            stats_df = pd.DataFrame({
                "Metric": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                "Value": [
                    f"{len(target_data):,}",
                    f"{target_data.mean():.4f}",
                    f"{target_data.std():.4f}",
                    f"{target_data.min():.4f}",
                    f"{target_data.quantile(0.25):.4f}",
                    f"{target_data.median():.4f}",
                    f"{target_data.quantile(0.75):.4f}",
                    f"{target_data.max():.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Distribution
            fig = px.histogram(
                df, 
                x=target_column, 
                title=f"Distribution of {target_column}",
                nbins=50
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        if time_column and time_column in df.columns:
            try:
                df_plot = df.copy()
                df_plot[time_column] = pd.to_datetime(df_plot[time_column])
                df_plot = df_plot.sort_values(time_column)
                
                fig = px.line(
                    df_plot, 
                    x=time_column, 
                    y=target_column,
                    title=f"{target_column} Over Time"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create time series plot: {str(e)}")
    
    # Data preprocessing options
    st.markdown('<div class="section-header">🧹 Data Preprocessing Options</div>', unsafe_allow_html=True)
    
    # Load saved preprocessing settings
    saved_prep = st.session_state.saved_settings['preprocessing']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🗑️ Data Cleaning:**")
        remove_nan = st.checkbox(
            "Remove NaN values", 
            value=saved_prep['remove_nan'], 
            help="Remove rows with missing values"
        )
        remove_negative = st.checkbox(
            "Remove negative values", 
            value=saved_prep['remove_negative'], 
            help="Remove rows with negative values in numeric columns"
        )
    
    with col2:
        st.markdown("**🎯 Outlier Removal:**")
        remove_outliers = st.checkbox(
            "Remove outliers", 
            value=saved_prep['remove_outliers'], 
            help="Remove statistical outliers"
        )
        
        if remove_outliers:
            outlier_methods = ["IQR (Interquartile Range)", "Mahalanobis Distance", "Both"]
            outlier_method = st.selectbox(
                "Outlier removal method:",
                outlier_methods,
                index=outlier_methods.index(saved_prep['outlier_method']) if saved_prep['outlier_method'] in outlier_methods else 0,
                help="Choose method for outlier detection"
            )
    
    # Store and update preprocessing options
    preprocessing_options = {
        "remove_nan": remove_nan,
        "remove_negative": remove_negative,
        "remove_outliers": remove_outliers,
        "outlier_method": outlier_method if remove_outliers else None
    }
    
    # Apply preprocessing using core modules
    if st.button("🔧 Apply Preprocessing", type="primary", use_container_width=True):
        if st.session_state.data is not None:
            try:
                with st.spinner("Applying preprocessing..."):
                    processed_data = apply_core_preprocessing(
                        st.session_state.raw_data.copy(), 
                        preprocessing_options
                    )
                    st.session_state.processed_data = processed_data
                    st.success(f"✅ Preprocessing applied! Data shape: {processed_data.shape}")
                    
                    # Show preprocessing summary
                    original_shape = st.session_state.raw_data.shape
                    new_shape = processed_data.shape
                    removed_rows = original_shape[0] - new_shape[0]
                    
                    if removed_rows > 0:
                        st.info(f"📉 Removed {removed_rows} rows ({removed_rows/original_shape[0]*100:.1f}%)")
                    
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
        else:
            st.warning("Please load a dataset first!")
    
    st.session_state.preprocessing_options = preprocessing_options
    st.session_state.saved_settings['preprocessing'] = preprocessing_options
    
    # Continue to next stage button
    if st.button("➡️ Continue to Feature Selection", type="primary"):
        st.session_state.stage = 2
        st.rerun()

def get_default_hyperparameters(model_name: str) -> dict:
    """Get default hyperparameter ranges for each model."""
    defaults = {
        "KNN": {
            "n_neighbors": {"min": 1, "max": 20, "default": [3, 5, 7]},
            "weights": {"options": ["uniform", "distance"], "default": ["uniform", "distance"]},
            "metric": {"options": ["euclidean", "manhattan", "minkowski"], "default": ["euclidean", "manhattan"]}
        },
        "RandomForest": {
            "n_estimators": {"min": 10, "max": 1000, "default": [100, 200, 300]},
            "max_depth": {"min": 1, "max": 50, "default": [10, 20, None]},
            "min_samples_split": {"min": 2, "max": 20, "default": [2, 5, 10]},
            "min_samples_leaf": {"min": 1, "max": 20, "default": [1, 2, 4]}
        },
        "XGBoost": {
            "n_estimators": {"min": 10, "max": 1000, "default": [100, 200, 300]},
            "learning_rate": {"min": 0.001, "max": 1.0, "default": [0.01, 0.1, 0.2]},
            "max_depth": {"min": 1, "max": 15, "default": [3, 6, 9]},
            "subsample": {"min": 0.1, "max": 1.0, "default": [0.8, 0.9, 1.0]}
        },
        "AdaBoost": {
            "n_estimators": {"min": 10, "max": 500, "default": [50, 100, 200]},
            "learning_rate": {"min": 0.01, "max": 2.0, "default": [0.1, 0.5, 1.0]}
        },
        "DecisionTree": {
            "max_depth": {"min": 1, "max": 50, "default": [10, 20, None]},
            "min_samples_split": {"min": 2, "max": 20, "default": [2, 5, 10]},
            "min_samples_leaf": {"min": 1, "max": 20, "default": [1, 2, 4]}
        },
        "DNN": {
            "hidden_layers": {"options": ["32", "64", "32,16", "64,32", "128,64", "128,64,32"], "default": ["64,32"]},
            "learning_rate": {"min": 0.0001, "max": 0.1, "default": [0.001, 0.01]},
            "dropout_rate": {"min": 0.0, "max": 0.8, "default": [0.2, 0.3, 0.5]},
            "batch_size": {"options": [16, 32, 64, 128], "default": [32, 64]}
        }
    }
    return defaults.get(model_name, {})

def stage_2_feature_selection():
    """Stage 2: Feature Selection (Placeholder for future implementation)."""
    if st.session_state.data is None:
        st.error("Please load a dataset first!")
        return
    
    st.markdown('<div class="section-header">2️⃣ Feature Selection</div>', unsafe_allow_html=True)
    
    # Placeholder content
    st.info("🚧 **Feature Selection Stage - Coming Soon!**")
    st.markdown("""
    This stage will include advanced feature selection techniques such as:
    
    - **📊 Statistical Tests**: Chi-square, ANOVA, correlation analysis
    - **🎯 Model-based Selection**: L1 regularization, tree-based importance
    - **🔄 Recursive Feature Elimination**: Backward/forward selection
    - **📈 Univariate Selection**: SelectKBest, SelectPercentile
    - **🧠 Mutual Information**: Information gain-based selection
    - **⚖️ Variance Threshold**: Remove low-variance features
    
    For now, all available numeric features will be used automatically.
    """)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Dataset Analysis"):
            st.session_state.stage = 1
            st.rerun()
    
    with col2:
        if st.button("➡️ Continue to Model Configuration", type="primary"):
            st.session_state.feature_selection_completed = True
            st.session_state.stage = 3
            st.rerun()

def stage_3_model_configuration():
    """Stage 3: Model Configuration."""
    if st.session_state.data is None:
        st.error("Please complete the previous stages first!")
        return
    
    st.markdown('<div class="section-header">3️⃣ Model Configuration</div>', unsafe_allow_html=True)
    
    # Load saved settings
    saved_settings = st.session_state.saved_settings
    
    # Initialize current config with saved values if not already set
    if not hasattr(st.session_state, 'current_model_config'):
        st.session_state.current_model_config = saved_settings['model_config'].copy()
    
    # Settings persistence and management
    st.markdown("### 💾 Settings Management")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info("� Your settings are automatically saved as you configure them.")
    
    with col2:
        if st.button("🔄 Reset All Settings", help="Reset all configuration to default values"):
            st.session_state.saved_settings = {
                'preprocessing': {
                    'remove_nan': True,
                    'remove_negative': True,
                    'remove_outliers': True,
                    'outlier_method': 'IQR (Interquartile Range)'
                },
                'model_config': {
                    'model': 'RandomForest',
                    'train_split': 0.8,
                    'scaler_type': 'StandardScaler',
                    'feature_engineering': {
                        'enabled': True,
                        'time_features': True,
                        'lag_features': True,
                        'lag_periods': [1, 4, 16],
                        'rolling_features': True,
                        'rolling_windows': [4, 24, 96]
                    },
                    'augmentation': {
                        'enabled': True,
                        'n_augment': 3,
                        'noise_level': 0.05
                    },
                    'hyperparameters': {}
                }
            }
            st.success("✅ All settings reset to defaults!")
            st.rerun()
    
    with col3:
        if st.button("📋 View Current Settings", help="Show all current configuration values"):
            with st.expander("Current Configuration", expanded=True):
                st.json(st.session_state.saved_settings)
    
    st.markdown("---")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    
    available_models = ["KNN", "RandomForest", "XGBoost", "AdaBoost", "DecisionTree", "DNN"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_config = st.session_state.current_model_config
        
        selected_model = st.selectbox(
            "Choose Model:",
            available_models,
            index=available_models.index(current_config['model']) if current_config['model'] in available_models else 0,
            help="Select the machine learning model to train",
            key="model_select"
        )
        
        train_split = st.slider(
            "Training/Test Split (%)",
            min_value=50,
            max_value=95,
            value=int(current_config['train_split'] * 100),
            help="Percentage of data used for training",
            key="train_split_slider"
        ) / 100
        
        # Feature scaling option
        scaler_options = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        scaler_type = st.selectbox(
            "Feature Scaling Method:",
            scaler_options,
            index=scaler_options.index(current_config['scaler_type']) if current_config['scaler_type'] in scaler_options else 0,
            help="Choose the method for scaling features",
            key="scaler_select"
        )
    
    with col2:
        st.markdown("**📋 Model Descriptions:**")
        model_descriptions = {
            "KNN": "K-Nearest Neighbors - Simple, interpretable",
            "RandomForest": "Random Forest - Robust, good performance", 
            "XGBoost": "Gradient Boosting - High performance",
            "AdaBoost": "Adaptive Boosting - Sequential learning",
            "DecisionTree": "Decision Tree - Highly interpretable",
            "DNN": "Deep Neural Network - Complex patterns"
        }
        
        for model, desc in model_descriptions.items():
            icon = "🔥" if model == selected_model else "•"
            st.write(f"{icon} **{model}**: {desc}")
        
        # Scaler descriptions
        st.markdown("**⚖️ Scaler Descriptions:**")
        scaler_descriptions = {
            "StandardScaler": "Mean=0, Std=1 (Normal distribution)",
            "MinMaxScaler": "Scale to [0,1] range",
            "RobustScaler": "Median=0, IQR=1 (Robust to outliers)"
        }
        
        for scaler, desc in scaler_descriptions.items():
            icon = "🔥" if scaler == scaler_type else "•"
            st.write(f"{icon} **{scaler}**: {desc}")
    
    # Hyperparameter Configuration
    st.markdown("### ⚙️ Hyperparameter Configuration")
    
    # Get default hyperparameters for selected model
    default_hyperparams = get_default_hyperparameters(selected_model)
    configured_hyperparams = {}
    
    if default_hyperparams:
        st.markdown(f"**🎛️ Configure {selected_model} Hyperparameters:**")
        
        for param_name, param_config in default_hyperparams.items():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**{param_name}:**")
            
            with col2:
                if "options" in param_config:
                    # Categorical parameter
                    configured_hyperparams[param_name] = st.multiselect(
                        f"Select {param_name}:",
                        param_config["options"],
                        default=param_config["default"],
                        key=f"{selected_model}_{param_name}"
                    )
                elif "min" in param_config and "max" in param_config:
                    # Numerical parameter
                    if isinstance(param_config["default"][0], int):
                        # Integer parameter
                        values = st.text_input(
                            f"Enter {param_name} values (comma-separated):",
                            value=", ".join(map(str, param_config["default"])),
                            key=f"{selected_model}_{param_name}",
                            help=f"Range: {param_config['min']} - {param_config['max']}"
                        )
                        try:
                            configured_hyperparams[param_name] = [int(x.strip()) for x in values.split(",") if x.strip()]
                        except:
                            configured_hyperparams[param_name] = param_config["default"]
                    else:
                        # Float parameter
                        values = st.text_input(
                            f"Enter {param_name} values (comma-separated):",
                            value=", ".join(map(str, param_config["default"])),
                            key=f"{selected_model}_{param_name}",
                            help=f"Range: {param_config['min']} - {param_config['max']}"
                        )
                        try:
                            configured_hyperparams[param_name] = [float(x.strip()) for x in values.split(",") if x.strip()]
                        except:
                            configured_hyperparams[param_name] = param_config["default"]

    # Feature engineering options
    st.markdown("### 🔧 Feature Engineering")
    
    current_fe = current_config['feature_engineering']
    enable_feature_engineering = st.checkbox(
        "Enable Feature Engineering", 
        value=current_fe['enabled'], 
        help="Create additional features from existing data",
        key="enable_fe"
    )
    
    if enable_feature_engineering:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏰ Time Features:**")
            time_features = st.checkbox(
                "Cyclic time features", 
                value=current_fe['time_features'], 
                help="Hour of day, day of year",
                key="time_features"
            )
        
        with col2:
            st.markdown("**📈 Lag Features:**")
            lag_features = st.checkbox(
                "Lag features", 
                value=current_fe['lag_features'], 
                help="Previous time step values",
                key="lag_features"
            )
            if lag_features:
                lag_periods = st.multiselect(
                    "Lag periods:",
                    [1, 2, 4, 8, 16, 24, 48],
                    default=current_fe['lag_periods'],
                    help="Number of time steps to look back",
                    key="lag_periods"
                )
        
        with col3:
            st.markdown("**📊 Rolling Features:**")
            rolling_features = st.checkbox(
                "Rolling window features", 
                value=current_fe['rolling_features'], 
                help="Moving averages and statistics",
                key="rolling_features"
            )
            if rolling_features:
                rolling_windows = st.multiselect(
                    "Window sizes:",
                    [4, 8, 12, 24, 48, 96],
                    default=current_fe['rolling_windows'],
                    help="Rolling window sizes",
                    key="rolling_windows"
                )
    
    # Data augmentation
    st.markdown("### 🔄 Data Augmentation")
    
    current_aug = current_config['augmentation']
    enable_augmentation = st.checkbox(
        "Enable Data Augmentation", 
        value=current_aug['enabled'], 
        help="Add artificial noise to increase training data",
        key="enable_augmentation"
    )
    
    if enable_augmentation:
        col1, col2 = st.columns(2)
        
        with col1:
            n_augment = st.slider(
                "Number of augmented copies", 
                1, 10, 
                current_aug['n_augment'], 
                help="How many noisy copies to create",
                key="n_augment"
            )
        
        with col2:
            noise_level = st.slider(
                "Noise level", 
                0.01, 0.2, 
                current_aug['noise_level'], 
                step=0.01, 
                help="Standard deviation of added noise",
                key="noise_level"
            )
    
    # Update current config with all values
    current_config['model'] = selected_model
    current_config['train_split'] = train_split  
    current_config['scaler_type'] = scaler_type
    current_config['hyperparameters'] = configured_hyperparams
    current_config['feature_engineering']['enabled'] = enable_feature_engineering
    if enable_feature_engineering:
        current_config['feature_engineering']['time_features'] = time_features
        current_config['feature_engineering']['lag_features'] = lag_features
        if lag_features:
            current_config['feature_engineering']['lag_periods'] = lag_periods
        else:
            current_config['feature_engineering']['lag_periods'] = []
        current_config['feature_engineering']['rolling_features'] = rolling_features
        if rolling_features:
            current_config['feature_engineering']['rolling_windows'] = rolling_windows
        else:
            current_config['feature_engineering']['rolling_windows'] = []
    else:
        current_config['feature_engineering']['time_features'] = False
        current_config['feature_engineering']['lag_features'] = False
        current_config['feature_engineering']['lag_periods'] = []
        current_config['feature_engineering']['rolling_features'] = False
        current_config['feature_engineering']['rolling_windows'] = []
    
    current_config['augmentation']['enabled'] = enable_augmentation
    if enable_augmentation:
        current_config['augmentation']['n_augment'] = n_augment
        current_config['augmentation']['noise_level'] = noise_level
    else:
        current_config['augmentation']['n_augment'] = 0
        current_config['augmentation']['noise_level'] = 0.0
    
    # Store configuration
    config = {
        "model": selected_model,
        "train_split": train_split,
        "scaler_type": scaler_type,
        "hyperparameters": configured_hyperparams,
        "feature_engineering": current_config['feature_engineering'],
        "augmentation": current_config['augmentation']
    }
    
    # Update session state
    st.session_state.current_model_config = current_config
    st.session_state.model_config = config
    
    # Configuration summary
    st.markdown("### 📋 Configuration Summary")
    
    with st.expander("View Configuration", expanded=False):
        st.json(config)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Feature Selection"):
            st.session_state.stage = 2
            st.rerun()
    
    with col2:
        if st.button("🚀 Start Training", type="primary"):
            st.session_state.stage = 4
            st.rerun()

def stage_4_training_results():
    """Stage 4: Training and Results."""
    if st.session_state.data is None or not hasattr(st.session_state, 'model_config'):
        st.error("Please complete the previous stages first!")
        return
    
    st.markdown('<div class="section-header">4️⃣ Training & Results</div>', unsafe_allow_html=True)
    
    # Training controls
    if st.session_state.model_results is None and not st.session_state.training_in_progress:
        # Show training button
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🎯 Start Model Training", type="primary", use_container_width=True):
                st.session_state.training_in_progress = True
                st.session_state.training_cancelled = False
                st.rerun()
        
        with col2:
            if st.button("⬅️ Back to Configuration", use_container_width=True):
                st.session_state.stage = 3
                st.rerun()
    
    elif st.session_state.training_in_progress and st.session_state.model_results is None:
        # Training in progress - show cancel button and progress
        st.markdown("### 🔄 Training in Progress...")
        
        # Progress information
        progress_container = st.container()
        with progress_container:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training steps simulation
                status_text.text("Initializing training...")
                progress_bar.progress(10)
                
            with col2:
                # Cancel button
                if st.button("❌ Cancel Training", type="secondary", use_container_width=True):
                    st.session_state.training_cancelled = True
                    st.session_state.training_in_progress = False
                    st.warning("Training cancelled by user!")
                    st.rerun()
        
        # Check if cancelled
        if not st.session_state.training_cancelled:
            try:
                # Update progress during training
                status_text.text("Preprocessing data...")
                progress_bar.progress(25)
                
                status_text.text("Training model...")
                progress_bar.progress(50)
                
                # Actually train the model
                results = train_and_evaluate_model()
                
                if not st.session_state.training_cancelled:
                    status_text.text("Evaluating results...")
                    progress_bar.progress(85)
                    
                    status_text.text("Finalizing...")
                    progress_bar.progress(100)
                    
                    st.session_state.model_results = results
                    st.session_state.training_in_progress = False
                    st.success("✅ Training completed successfully!")
                    st.rerun()
                
            except Exception as e:
                st.session_state.training_in_progress = False
                if not st.session_state.training_cancelled:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
    
    elif st.session_state.model_results is not None:
        # Display results
        display_results()

def train_and_evaluate_model():
    """Train and evaluate the selected model."""
    # Get data and configuration
    df = st.session_state.data.copy()
    config = st.session_state.model_config
    preprocessing = st.session_state.preprocessing_options
    
    # Apply preprocessing
    if preprocessing["remove_nan"]:
        df = df.dropna()
    
    if preprocessing["remove_negative"]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mask = (df[numeric_cols] >= 0).all(axis=1)
        df = df[mask]
    
    # Basic outlier removal (simplified)
    if preprocessing["remove_outliers"]:
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != st.session_state.time_column]
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Prepare features and target
    target_col = st.session_state.target_column
    feature_cols = st.session_state.feature_columns
    time_col = st.session_state.time_column
    
    # Sort by time if time column exists
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    
    # Feature engineering (simplified)
    if config["feature_engineering"]["enabled"]:
        if config["feature_engineering"]["time_features"] and time_col in df.columns:
            df['hour'] = df[time_col].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['doy'] = df[time_col].dt.dayofyear
            df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)
            feature_cols.extend(['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos'])
        
        # Add lag features
        if config["feature_engineering"]["lag_features"]:
            for col in st.session_state.feature_columns:
                for lag in config["feature_engineering"]["lag_periods"]:
                    lag_col = f"{col}_lag{lag}"
                    df[lag_col] = df[col].shift(lag)
                    feature_cols.append(lag_col)
        
        # Add rolling features (simplified)
        if config["feature_engineering"]["rolling_features"]:
            for col in st.session_state.feature_columns:
                for window in config["feature_engineering"]["rolling_windows"]:
                    roll_mean_col = f"{col}_roll{window}_mean"
                    roll_std_col = f"{col}_roll{window}_std"
                    df[roll_mean_col] = df[col].rolling(window=window).mean()
                    df[roll_std_col] = df[col].rolling(window=window).std()
                    feature_cols.extend([roll_mean_col, roll_std_col])
    
    # Remove rows with NaN values created by feature engineering
    df = df.dropna().reset_index(drop=True)
    
    # Split data
    split_idx = int(len(df) * config["train_split"])
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Prepare features and target
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Scale features with selected scaler
    scaler_type = config["scaler_type"]
    if scaler_type == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply data augmentation
    if config["augmentation"]["enabled"]:
        X_train_aug, y_train_aug = add_noise(
            X_train_scaled, y_train,
            n_augment=config["augmentation"]["n_augment"],
            noise_level=config["augmentation"]["noise_level"],
            random_state=42
        )
    else:
        X_train_aug, y_train_aug = X_train_scaled, y_train
    
    # Train model
    model_name = config["model"]
    
    # Use configured hyperparameters or fallback to defaults
    configured_hyperparams = config.get("hyperparameters", {})
    
    # Fallback hyperparameter grids if not configured
    default_param_grids = {
        "KNN": {"n_neighbors": [3, 5, 7]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.1, 1.0]},
        "DecisionTree": {"max_depth": [10, 20], "min_samples_split": [2, 5]},
        "DNN": {"hidden_layers": [64, 32], "learning_rate": 0.001}
    }
    
    # Use configured hyperparameters if available, otherwise use defaults
    if configured_hyperparams and model_name in ["KNN", "RandomForest", "XGBoost", "AdaBoost", "DecisionTree"]:
        param_grid = configured_hyperparams
    else:
        param_grid = default_param_grids.get(model_name, {})
    
    if model_name == "DNN":
        # Simplified DNN training
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_aug.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train_aug, y_train_aug, epochs=50, batch_size=32, verbose=0)
        
        y_train_pred = model.predict(X_train_scaled).flatten()
        y_test_pred = model.predict(X_test_scaled).flatten()
        best_params = {"architecture": "64-32-1", "learning_rate": 0.001}
    else:
        # Use sklearn models
        model, best_params = train_model(
            X_train_aug, y_train_aug, model_name,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='r2'
        )
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate results
    train_metrics = evaluate_regression(y_train, y_train_pred)
    test_metrics = evaluate_regression(y_test, y_test_pred)
    
    # Prepare results
    results = {
        "model_name": model_name,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_train_true": y_train,
        "y_train_pred": y_train_pred,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "feature_names": feature_cols,
        "train_size": len(X_train_aug),
        "test_size": len(X_test),
        "train_df": train_df,
        "test_df": test_df
    }
    
    return results

def display_results():
    """Display training and evaluation results."""
    results = st.session_state.model_results
    
    st.success(f"✅ {results['model_name']} model trained successfully!")
    
    # Performance metrics
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Training Metrics:**")
        train_metrics = results["train_metrics"]
        
        metrics_data = []
        for metric, value in train_metrics.items():
            metrics_data.append({"Metric": metric, "Value": f"{value:.4f}"})
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    with col2:
        st.markdown("**🧪 Test Metrics:**")
        test_metrics = results["test_metrics"]
        
        metrics_data = []
        for metric, value in test_metrics.items():
            metrics_data.append({"Metric": metric, "Value": f"{value:.4f}"})
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    # Model information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{results['train_size']:,}")
    with col2:
        st.metric("Test Samples", f"{results['test_size']:,}")
    with col3:
        st.metric("Features Used", len(results['feature_names']))
    
    # Best parameters
    with st.expander("🔧 Best Parameters"):
        st.json(results["best_params"])
    
    # Prediction vs True plots
    st.markdown("### 📈 Prediction Results")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Training: True vs Predicted", "Test: True vs Predicted", 
                       "Training Residuals", "Test Residuals"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training scatter plot
    fig.add_trace(
        go.Scatter(
            x=results["y_train_true"], 
            y=results["y_train_pred"],
            mode='markers',
            name='Training',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Test scatter plot
    fig.add_trace(
        go.Scatter(
            x=results["y_test_true"], 
            y=results["y_test_pred"],
            mode='markers',
            name='Test',
            marker=dict(color='red', opacity=0.6)
        ),
        row=1, col=2
    )
    
    # Training residuals
    train_residuals = results["y_train_true"] - results["y_train_pred"]
    fig.add_trace(
        go.Scatter(
            x=results["y_train_pred"], 
            y=train_residuals,
            mode='markers',
            name='Train Residuals',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=2, col=1
    )
    
    # Test residuals
    test_residuals = results["y_test_true"] - results["y_test_pred"]
    fig.add_trace(
        go.Scatter(
            x=results["y_test_pred"], 
            y=test_residuals,
            mode='markers',
            name='Test Residuals',
            marker=dict(color='red', opacity=0.6)
        ),
        row=2, col=2
    )
    
    # Add perfect prediction lines
    for row, col in [(1, 1), (1, 2)]:
        if row == 1 and col == 1:
            min_val, max_val = min(results["y_train_true"]), max(results["y_train_true"])
        else:
            min_val, max_val = min(results["y_test_true"]), max(results["y_test_true"])
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='gray'),
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
    
    # Add zero lines for residuals
    for col in [1, 2]:
        if col == 1:
            x_range = [min(results["y_train_pred"]), max(results["y_train_pred"])]
        else:
            x_range = [min(results["y_test_pred"]), max(results["y_test_pred"])]
        
        fig.add_trace(
            go.Scatter(
                x=x_range, 
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(dash='dash', color='gray'),
                showlegend=(col == 1)
            ),
            row=2, col=col
        )
    
    fig.update_layout(height=800, title_text="Model Performance Analysis")
    fig.update_xaxes(title_text="True Values", row=1, col=1)
    fig.update_xaxes(title_text="True Values", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
    fig.update_xaxes(title_text="Predicted Values", row=2, col=2)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series comparison
    if st.session_state.time_column in results["test_df"].columns:
        st.markdown("### ⏰ Time Series Comparison")
        
        # Create time series plot
        test_df = results["test_df"].copy()
        test_df["predicted"] = results["y_test_pred"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_df[st.session_state.time_column],
            y=results["y_test_true"],
            mode='lines',
            name='True Values',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_df[st.session_state.time_column],
            y=results["y_test_pred"],
            mode='lines',
            name='Predicted Values',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Time Series: True vs Predicted Values",
            xaxis_title="Time",
            yaxis_title=st.session_state.target_column,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    if hasattr(results.get("model"), "feature_importances_"):
        st.markdown("### 🎯 Feature Importance")
        
        importance_data = pd.DataFrame({
            "Feature": results["feature_names"],
            "Importance": results["model"].feature_importances_
        }).sort_values("Importance", ascending=False)
        
        fig = px.bar(
            importance_data.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Most Important Features"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predictions CSV
        pred_df = pd.DataFrame({
            "true_values": results["y_test_true"],
            "predicted_values": results["y_test_pred"],
            "residuals": results["y_test_true"] - results["y_test_pred"]
        })
        
        csv_data = pred_df.to_csv(index=False)
        st.download_button(
            "📄 Download Predictions (CSV)",
            csv_data,
            file_name=f"{results['model_name']}_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        # Model summary
        summary = {
            "model": results["model_name"],
            "best_parameters": results["best_params"],
            "training_metrics": results["train_metrics"],
            "test_metrics": results["test_metrics"],
            "feature_count": len(results["feature_names"]),
            "training_samples": results["train_size"],
            "test_samples": results["test_size"]
        }
        
        summary_json = yaml.dump(summary, default_flow_style=False)
        st.download_button(
            "📋 Download Summary (YAML)",
            summary_json,
            file_name=f"{results['model_name']}_summary.yaml",
            mime="text/yaml"
        )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Train Another Model", type="secondary", use_container_width=True):
            st.session_state.model_results = None
            st.session_state.training_in_progress = False
            st.session_state.training_cancelled = False
            st.session_state.stage = 3
            st.rerun()
    
    with col2:
        if st.button("📊 New Dataset", type="secondary", use_container_width=True):
            # Reset all session state
            for key in list(st.session_state.keys()):
                if key not in ['saved_settings']:  # Keep saved settings
                    del st.session_state[key]
            st.session_state.stage = 1
            st.rerun()
    
    with col3:
        if st.button("⚙️ Modify Configuration", type="secondary", use_container_width=True):
            st.session_state.model_results = None
            st.session_state.training_in_progress = False
            st.session_state.training_cancelled = False
            st.session_state.stage = 3
            st.rerun()

if __name__ == "__main__":
    main()