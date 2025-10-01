"""Plugin system for predictN2O pipeline.

This module provides a flexible plugin architecture that allows:
- Dynamic loading of feature engineering plugins
- Registration of custom preprocessing steps
- Extensible evaluation metrics and analysis tools

Plugin Types:
- FeatureEngineeringPlugin: For creating new features
- PreprocessingPlugin: For data cleaning and preparation
- EvaluationPlugin: For model analysis and metrics

Example usage:
    registry = PluginRegistry()
    registry.load_plugins("src/plugins/feature_engineering")
    
    # Use a plugin
    plugin = registry.get_plugin("lag_features")
    processed_data = plugin.transform(data, config)
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""
    pass


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    pass


# =============================================================================
# Base Plugin Classes
# =============================================================================

class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.enabled = True
        
    @abstractmethod
    def execute(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Execute the plugin's main functionality.
        
        Args:
            data: Input data to process
            config: Configuration parameters
            **kwargs: Additional arguments
            
        Returns:
            Processed data
        """
        pass
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
        
    def get_dependencies(self) -> List[str]:
        """Get list of required dependencies.
        
        Returns:
            List of dependency names
        """
        return []
        
    def __str__(self) -> str:
        return f"{self.name} v{self.version}"


class FeatureEngineeringPlugin(BasePlugin):
    """Base class for feature engineering plugins."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform input DataFrame by adding/modifying features.
        
        Args:
            df: Input DataFrame
            config: Feature engineering configuration
            
        Returns:
            Transformed DataFrame
        """
        pass
        
    def execute(self, data: pd.DataFrame, config: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Execute feature engineering transformation."""
        return self.transform(data, config)
        
    def get_feature_names(self, input_features: List[str], config: Dict[str, Any]) -> List[str]:
        """Get names of features that will be created.
        
        Args:
            input_features: List of input feature names
            config: Configuration
            
        Returns:
            List of new feature names
        """
        return []


class PreprocessingPlugin(BasePlugin):
    """Base class for data preprocessing plugins."""
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input DataFrame.
        
        Args:
            df: Input DataFrame
            config: Preprocessing configuration
            
        Returns:
            Preprocessed DataFrame
        """
        pass
        
    def execute(self, data: pd.DataFrame, config: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Execute preprocessing."""
        return self.preprocess(data, config)


class EvaluationPlugin(BasePlugin):
    """Base class for evaluation plugins."""
    
    @abstractmethod
    def evaluate(self, y_true: Any, y_pred: Any, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model: Trained model
            config: Evaluation configuration
            
        Returns:
            Dictionary of evaluation results
        """
        pass
        
    def execute(self, data: Any, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute evaluation."""
        y_true = kwargs.get('y_true')
        y_pred = kwargs.get('y_pred')
        model = kwargs.get('model')
        
        if y_true is None or y_pred is None:
            raise PluginExecutionError("y_true and y_pred must be provided for evaluation plugins")
            
        return self.evaluate(y_true, y_pred, model, config)


# =============================================================================
# Plugin Registry
# =============================================================================

class PluginRegistry:
    """Registry for managing and executing plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_types: Dict[str, Type[BasePlugin]] = {
            'feature_engineering': FeatureEngineeringPlugin,
            'preprocessing': PreprocessingPlugin,
            'evaluation': EvaluationPlugin,
        }
        
    def register_plugin(self, plugin: BasePlugin, plugin_type: str = None) -> None:
        """Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
            plugin_type: Type of plugin (optional, auto-detected)
            
        Raises:
            PluginRegistrationError: If registration fails
        """
        if not isinstance(plugin, BasePlugin):
            raise PluginRegistrationError(f"Plugin must inherit from BasePlugin: {plugin}")
            
        if plugin.name in self._plugins:
            logger.warning(f"Overwriting existing plugin: {plugin.name}")
            
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin}")
        
    def load_plugins_from_directory(self, directory: str, plugin_type: str = None) -> int:
        """Load all plugins from a directory.
        
        Args:
            directory: Directory path containing plugin files
            plugin_type: Expected plugin type (optional)
            
        Returns:
            Number of plugins loaded
            
        Raises:
            PluginError: If loading fails
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return 0
            
        loaded_count = 0
        
        # Find all Python files in directory
        for py_file in directory_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                loaded_count += self._load_plugin_from_file(py_file, plugin_type)
            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}")
                
        logger.info(f"Loaded {loaded_count} plugins from {directory}")
        return loaded_count
        
    def _load_plugin_from_file(self, file_path: Path, expected_type: str = None) -> int:
        """Load plugin(s) from a single Python file.
        
        Args:
            file_path: Path to Python file
            expected_type: Expected plugin type
            
        Returns:
            Number of plugins loaded from file
        """
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            raise PluginError(f"Could not load module spec from {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        loaded_count = 0
        
        # Find plugin classes in module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != BasePlugin and 
                issubclass(obj, BasePlugin) and 
                obj.__module__ == module.__name__):
                
                try:
                    # Instantiate plugin
                    plugin_instance = obj()
                    self.register_plugin(plugin_instance, expected_type)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {name}: {e}")
                    
        return loaded_count
        
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
        
    def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> List[BasePlugin]:
        """Get all plugins of a specific type.
        
        Args:
            plugin_type: Plugin type class
            
        Returns:
            List of matching plugins
        """
        return [plugin for plugin in self._plugins.values() 
                if isinstance(plugin, plugin_type)]
        
    def list_plugins(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins.
        
        Returns:
            Dictionary of plugin name -> plugin instance
        """
        return self._plugins.copy()
        
    def execute_plugin(self, name: str, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Execute a plugin by name.
        
        Args:
            name: Plugin name
            data: Input data
            config: Configuration
            **kwargs: Additional arguments
            
        Returns:
            Plugin execution result
            
        Raises:
            PluginError: If plugin not found or execution fails
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise PluginError(f"Plugin not found: {name}")
            
        if not plugin.enabled:
            logger.warning(f"Plugin {name} is disabled, skipping execution")
            return data
            
        try:
            logger.debug(f"Executing plugin: {name}")
            return plugin.execute(data, config, **kwargs)
        except Exception as e:
            raise PluginExecutionError(f"Plugin {name} execution failed: {e}")
            
    def execute_plugin_chain(self, plugin_names: List[str], data: Any, 
                           config: Dict[str, Any], **kwargs) -> Any:
        """Execute a chain of plugins in sequence.
        
        Args:
            plugin_names: List of plugin names to execute
            data: Initial input data
            config: Configuration
            **kwargs: Additional arguments
            
        Returns:
            Final result after all plugins
        """
        result = data
        
        for plugin_name in plugin_names:
            result = self.execute_plugin(plugin_name, result, config, **kwargs)
            
        return result
        
    def validate_plugins(self, config: Dict[str, Any]) -> List[str]:
        """Validate all registered plugins.
        
        Args:
            config: Configuration to validate against
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for name, plugin in self._plugins.items():
            try:
                if not plugin.validate_config(config):
                    errors.append(f"Plugin {name} configuration validation failed")
            except Exception as e:
                errors.append(f"Plugin {name} validation error: {e}")
                
        return errors
        
    def print_plugin_summary(self) -> None:
        """Print summary of registered plugins."""
        print(f"\n{'='*60}")
        print("REGISTERED PLUGINS")
        print(f"{'='*60}")
        
        if not self._plugins:
            print("No plugins registered.")
            return
            
        # Group by type
        plugin_groups = {}
        for plugin in self._plugins.values():
            plugin_type = type(plugin).__name__
            if plugin_type not in plugin_groups:
                plugin_groups[plugin_type] = []
            plugin_groups[plugin_type].append(plugin)
            
        for plugin_type, plugins in plugin_groups.items():
            print(f"\n{plugin_type}:")
            for plugin in plugins:
                status = "✓" if plugin.enabled else "✗"
                print(f"  {status} {plugin.name} v{plugin.version}")
                if plugin.description:
                    print(f"    {plugin.description}")
                    
        print(f"\nTotal: {len(self._plugins)} plugins")
        print(f"{'='*60}\n")


# Global plugin registry
plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return plugin_registry


def register_plugin(plugin: BasePlugin) -> None:
    """Register a plugin with the global registry."""
    plugin_registry.register_plugin(plugin)


def load_plugins(config: Dict[str, Any]) -> None:
    """Load plugins based on configuration.
    
    Args:
        config: Configuration containing plugin settings
    """
    plugins_config = config.get('plugins', {})
    
    for plugin_type, type_config in plugins_config.items():
        if not type_config.get('enabled_plugins'):
            continue
            
        search_paths = type_config.get('plugin_search_paths', [])
        for search_path in search_paths:
            plugin_registry.load_plugins_from_directory(search_path, plugin_type)
            
    # Validate loaded plugins
    errors = plugin_registry.validate_plugins(config)
    if errors:
        logger.warning(f"Plugin validation warnings:\n" + "\n".join(f"  - {error}" for error in errors))