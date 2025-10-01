"""Configuration management system for predictN2O pipeline.

This module provides a flexible configuration system that supports:
- YAML configuration files
- CLI argument overrides
- Environment variable overrides
- Configuration validation
- Default fallbacks

Example usage:
    config = ConfigManager()
    config.load_config("config/my_config.yaml")
    config.override_from_cli(args)
    
    # Access nested configuration
    model_name = config.get("models.default_model")
    periods = config.get("periods")
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    message: str
    path: str = ""


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, default_config_path: Optional[str] = None):
        """Initialize ConfigManager.
        
        Args:
            default_config_path: Path to default configuration file
        """
        self._config = {}
        self._default_config_path = default_config_path or "config/default_config.yaml"
        self._loaded_configs = []
        
    def load_default_config(self) -> None:
        """Load the default configuration."""
        if os.path.exists(self._default_config_path):
            self.load_config(self._default_config_path)
        else:
            logger.warning(f"Default config file not found: {self._default_config_path}")
            
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
                
            if new_config is None:
                logger.warning(f"Empty configuration file: {config_path}")
                return
                
            # Deep merge with existing configuration
            self._config = self._deep_merge(self._config, new_config)
            self._loaded_configs.append(str(config_path))
            
            logger.info(f"Loaded configuration from: {config_path}")
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
            
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base_dict)
        
        for key, value in update_dict.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
                
        return result
        
    def override_from_dict(self, overrides: Dict[str, Any]) -> None:
        """Override configuration values from dictionary.
        
        Args:
            overrides: Dictionary with override values
                      Keys can be nested using dot notation (e.g., "models.default_model")
        """
        for key, value in overrides.items():
            self.set(key, value)
            
    def override_from_cli(self, args) -> None:
        """Override configuration from CLI arguments.
        
        Args:
            args: Parsed CLI arguments (argparse.Namespace)
        """
        # Map CLI arguments to config paths
        cli_mappings = {
            'model': 'models.default_model',
            'split': 'data.default_split_ratio',
            'augment': 'augmentation.n_augment',
            'noise': 'augmentation.noise_level',
            'seed': 'general.random_seed',
            'verbose': 'general.verbose',
        }
        
        for cli_arg, config_path in cli_mappings.items():
            if hasattr(args, cli_arg):
                value = getattr(args, cli_arg)
                if value is not None:
                    self.set(config_path, value)
                    logger.info(f"CLI override: {config_path} = {value}")
                    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            path: Dot-separated path to configuration value (e.g., "models.default_model")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        current = self._config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
            
    def set(self, path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = path.split('.')
        current = self._config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # Set the final value
        current[keys[-1]] = value
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., "models")
            
        Returns:
            Dictionary containing the section
        """
        return self.get(section, {})
        
    def validate(self) -> None:
        """Validate configuration integrity.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        self._validate_required_sections()
        self._validate_paths()
        self._validate_models()
        self._validate_periods()
        
    def _validate_required_sections(self) -> None:
        """Validate that required sections exist."""
        required_sections = [
            'general', 'data', 'periods', 'models', 
            'feature_engineering', 'output', 'logging'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigValidationError(
                    f"Required configuration section missing: {section}",
                    section
                )
                
    def _validate_paths(self) -> None:
        """Validate that required paths exist or can be created."""
        # Check data file exists
        data_path = self.get('data.data_path')
        if data_path and not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            
        # Ensure output directories can be created
        output_dirs = [
            'output.results_dir',
            'output.model_output_dir', 
            'output.figures_dir',
            'output.logs_dir',
            'output.predictions_dir'
        ]
        
        for dir_path_key in output_dirs:
            dir_path = self.get(dir_path_key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                
    def _validate_models(self) -> None:
        """Validate model configuration."""
        available_models = self.get('models.available_models', [])
        default_model = self.get('models.default_model')
        
        if default_model and default_model not in available_models:
            raise ConfigValidationError(
                f"Default model '{default_model}' not in available models: {available_models}",
                "models.default_model"
            )
            
    def _validate_periods(self) -> None:
        """Validate period configuration."""
        periods = self.get('periods', [])
        
        if not periods:
            raise ConfigValidationError("No periods defined", "periods")
            
        required_period_fields = ['name', 'start_date', 'end_date']
        for i, period in enumerate(periods):
            for field in required_period_fields:
                if field not in period:
                    raise ConfigValidationError(
                        f"Period {i} missing required field: {field}",
                        f"periods[{i}].{field}"
                    )
                    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return deepcopy(self._config)
        
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to: {output_path}")
        
    def print_summary(self) -> None:
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print("CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        
        if self._loaded_configs:
            print(f"Loaded configs: {', '.join(self._loaded_configs)}")
        
        # Key settings
        key_settings = [
            ("Project", "general.project_name"),
            ("Default Model", "models.default_model"),
            ("Data Path", "data.data_path"),
            ("Random Seed", "general.random_seed"),
            ("Split Ratio", "data.default_split_ratio"),
            ("Results Dir", "output.results_dir"),
        ]
        
        print("\nKey Settings:")
        for label, path in key_settings:
            value = self.get(path, "Not Set")
            print(f"  {label}: {value}")
            
        # Periods
        periods = self.get('periods', [])
        print(f"\nConfigured Periods: {len(periods)}")
        for period in periods:
            print(f"  - {period.get('name', 'Unknown')}: {period.get('start_date', '')} to {period.get('end_date', '')}")
            
        # Models
        models = self.get('models.available_models', [])
        print(f"\nAvailable Models: {', '.join(models)}")
        
        print(f"{'='*60}\n")


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    return config


def load_config(config_path: str) -> ConfigManager:
    """Load configuration and return the global instance."""
    config.load_config(config_path)
    return config


def init_config(config_path: Optional[str] = None, cli_args=None) -> ConfigManager:
    """Initialize configuration with defaults, file, and CLI overrides.
    
    Args:
        config_path: Optional path to configuration file
        cli_args: Optional CLI arguments to override config
        
    Returns:
        Configured ConfigManager instance
    """
    # Load default configuration
    config.load_default_config()
    
    # Load additional configuration file if provided
    if config_path and config_path != config._default_config_path:
        config.load_config(config_path)
        
    # Apply CLI overrides
    if cli_args:
        config.override_from_cli(cli_args)
        
    # Validate configuration
    config.validate()
    
    return config