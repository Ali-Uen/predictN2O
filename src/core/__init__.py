"""Core modules package initialization."""

from .config_manager import ConfigManager, get_config, load_config, init_config
from .plugin_system import PluginRegistry, get_plugin_registry, register_plugin, load_plugins

__all__ = [
    'ConfigManager',
    'get_config', 
    'load_config',
    'init_config',
    'PluginRegistry',
    'get_plugin_registry',
    'register_plugin',
    'load_plugins'
]