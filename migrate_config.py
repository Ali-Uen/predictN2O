"""Migration script to convert legacy configuration to new YAML format.

This script extracts configuration from existing Python modules and creates
YAML configuration files compatible with the new system.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.config import (
        DATA_PATH, PERIODS, DEFAULT_MODEL, DEFAULT_SPLIT_RATIO, RANDOM_SEED,
        AUGMENTATION_N, AUGMENTATION_NOISE, RESULTS_DIR, SAVE_RESULTS, SAVE_FIGURES,
        VERBOSE, MODELS
    )
    from src.param_grids import param_grids
    from src.resampling import resampling
    from src.feature_engineering import model_period_lags, model_period_rolling
except ImportError as e:
    print(f"Error importing legacy configuration: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def migrate_configuration():
    """Create YAML configuration from legacy Python modules."""
    
    print("Migrating legacy configuration to YAML format...")
    
    # Build configuration dictionary
    config = {
        'general': {
            'project_name': 'predictN2O',
            'random_seed': RANDOM_SEED,
            'verbose': VERBOSE,
            'save_results': SAVE_RESULTS,
            'save_figures': SAVE_FIGURES
        },
        
        'data': {
            'data_path': DATA_PATH,
            'time_column': 'TIME',
            'target_column': 'N2O',
            'feature_columns': ['DO', 'T', 'Q_in'],
            'exclude_cols': ['TIME'],
            'include_cols': None,
            'default_split_ratio': DEFAULT_SPLIT_RATIO,
            'min_samples_period': 50,
            'min_test_samples': 10
        },
        
        'periods': [],
        
        'models': {
            'available_models': MODELS,
            'default_model': DEFAULT_MODEL,
            'cv_folds': 5,
            'cv_scoring': 'r2',
            'cv_n_jobs': -1,
            'cv_verbose': 2
        },
        
        'feature_engineering': {
            'time_features': {
                'enabled': True,
                'cyclic_hour': True,
                'cyclic_day_of_year': True
            },
            'lag_features': {
                'enabled': True,
                'default_lags': {
                    'DO': [1, 2],
                    'Q_in': [1, 2],
                    'T': [1, 2]
                }
            },
            'rolling_features': {
                'enabled': True,
                'default_windows': {
                    'DO': [3, 6, 12, 24],
                    'Q_in': [3, 6, 12, 24],
                    'T': [3, 6, 12, 24]
                },
                'statistics': ['mean', 'std']
            }
        },
        
        'augmentation': {
            'enabled': AUGMENTATION_N > 0,
            'n_augment': AUGMENTATION_N,
            'noise_level': AUGMENTATION_NOISE
        },
        
        'resampling': {
            'default_rule': None,
            'model_overrides': {}
        },
        
        'output': {
            'results_dir': RESULTS_DIR,
            'model_output_dir': f"{RESULTS_DIR}/model_outputs",
            'figures_dir': f"{RESULTS_DIR}/figures",
            'logs_dir': f"{RESULTS_DIR}/logs",
            'predictions_dir': f"{RESULTS_DIR}/predictions",
            'model_format': 'joblib',
            'results_format': 'csv'
        },
        
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_logging': True,
            'console_logging': True,
            'log_file': f"{RESULTS_DIR}/logs/pipeline.log"
        },
        
        'hyperparameters': {},
        
        'plugins': {
            'feature_engineering': {
                'enabled_plugins': ['time_features', 'lag_features', 'rolling_features'],
                'plugin_search_paths': ['src/plugins/feature_engineering']
            }
        }
    }
    
    # Convert periods
    for name, start_date, end_date in PERIODS:
        config['periods'].append({
            'name': name,
            'start_date': start_date,
            'end_date': end_date,
            'description': f"Period from {start_date} to {end_date}"
        })
    
    # Convert hyperparameter grids
    for model_name, model_grids in param_grids.items():
        # Take first period's parameters as default (they're all the same in current config)
        first_period = list(model_grids.keys())[0]
        config['hyperparameters'][model_name] = model_grids[first_period]
    
    # Convert resampling rules
    for model_name, model_resampling in resampling.items():
        for period_name, rule in model_resampling.items():
            if rule is not None:
                if model_name not in config['resampling']['model_overrides']:
                    config['resampling']['model_overrides'][model_name] = {}
                config['resampling']['model_overrides'][model_name][period_name] = rule
    
    # Add legacy model/period specific configurations for backward compatibility
    config['model_period_lags'] = model_period_lags
    config['model_period_rolling'] = model_period_rolling
    
    return config


def save_migrated_config(config, output_path='config/migrated_config.yaml'):
    """Save migrated configuration to YAML file."""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"✅ Migrated configuration saved to: {output_path}")


def create_example_configs():
    """Create example configuration files for different use cases."""
    
    base_config = migrate_configuration()
    
    # Example 1: Experiment with different models
    experiment_config = base_config.copy()
    experiment_config['general']['project_name'] = 'N2O_Model_Comparison'
    experiment_config['augmentation']['enabled'] = True
    experiment_config['augmentation']['n_augment'] = 3
    experiment_config['augmentation']['noise_level'] = 0.05
    
    save_migrated_config(experiment_config, 'config/model_experiment.yaml')
    
    # Example 2: Quick test configuration
    test_config = base_config.copy()
    test_config['general']['project_name'] = 'N2O_Quick_Test'
    test_config['models']['default_model'] = 'KNN'
    test_config['models']['cv_folds'] = 3
    test_config['periods'] = [base_config['periods'][0]]  # Only first period
    
    save_migrated_config(test_config, 'config/quick_test.yaml')
    
    # Example 3: Production configuration
    prod_config = base_config.copy()
    prod_config['general']['project_name'] = 'N2O_Production'
    prod_config['logging']['level'] = 'WARNING'
    prod_config['general']['verbose'] = False
    
    save_migrated_config(prod_config, 'config/production.yaml')
    
    print("✅ Example configurations created:")
    print("  - config/model_experiment.yaml (with augmentation)")
    print("  - config/quick_test.yaml (fast testing)")
    print("  - config/production.yaml (minimal logging)")


def validate_migration():
    """Validate that migration preserves key settings."""
    
    print("\n🔍 Validating migration...")
    
    # Load migrated config
    with open('config/migrated_config.yaml', 'r') as f:
        migrated = yaml.safe_load(f)
    
    # Check key values
    checks = [
        (migrated['general']['random_seed'] == RANDOM_SEED, "Random seed"),
        (migrated['data']['default_split_ratio'] == DEFAULT_SPLIT_RATIO, "Split ratio"),
        (migrated['models']['default_model'] == DEFAULT_MODEL, "Default model"),
        (len(migrated['periods']) == len(PERIODS), "Number of periods"),
        (migrated['data']['data_path'] == DATA_PATH, "Data path"),
    ]
    
    all_passed = True
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {description}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ Migration validation successful!")
    else:
        print("\n❌ Migration validation failed - please check configuration")
    
    return all_passed


def main():
    """Main migration script."""
    
    print("=" * 60)
    print("PredictN2O Configuration Migration")
    print("=" * 60)
    
    try:
        # Migrate configuration
        config = migrate_configuration()
        
        # Save main migrated config
        save_migrated_config(config, 'config/migrated_config.yaml')
        
        # Create example configs
        create_example_configs()
        
        # Validate migration
        if validate_migration():
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("1. Test the new system: python main_modern.py --config config/migrated_config.yaml --dry-run")
            print("2. Run a quick test: python main_modern.py --config config/quick_test.yaml")
            print("3. Compare with original: python main.py")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())