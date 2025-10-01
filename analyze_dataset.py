#!/usr/bin/env python3
"""
Dataset Analysis and Configuration Generator

This command-line tool analyzes any dataset and automatically generates
appropriate configuration files for the predictN2O pipeline.

Usage examples:
    python analyze_dataset.py data/my_dataset.csv
    python analyze_dataset.py data/sensor_data.csv --output config/sensor_config.yaml
    python analyze_dataset.py data/emissions.csv --interactive
    python analyze_dataset.py data/wastewater.csv --generate-variants
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.dataset_analyzer import DatasetAnalyzer, analyze_and_configure
import yaml
import logging


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def interactive_configuration(analysis_result: dict) -> dict:
    """Interactive configuration refinement."""
    print(f"\n{'='*60}")
    print("INTERACTIVE CONFIGURATION")
    print(f"{'='*60}")
    
    config = analysis_result['recommended_config'].copy()
    
    # Confirm target column
    target_suggestions = analysis_result['target_suggestions']['candidates']
    if target_suggestions:
        print(f"\nTarget column suggestions:")
        for i, candidate in enumerate(target_suggestions[:5]):
            print(f"  {i+1}. {candidate['column']} (confidence: {candidate['confidence']:.2f})")
        
        choice = input(f"\nSelect target column [1-{len(target_suggestions[:5])}] or enter custom name: ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(target_suggestions):
                config['data']['target_column'] = target_suggestions[choice_idx]['column']
        except ValueError:
            if choice:
                config['data']['target_column'] = choice
    
    # Confirm feature columns
    recommended_features = analysis_result['columns']['recommended_features']
    print(f"\nRecommended feature columns ({len(recommended_features)}):")
    for i, col in enumerate(recommended_features):
        print(f"  {i+1}. {col}")
    
    keep_features = input(f"\nUse all recommended features? [Y/n]: ").strip().lower()
    if keep_features in ['n', 'no']:
        selected_indices = input("Enter feature numbers to keep (e.g., 1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
            selected_features = [recommended_features[i] for i in indices if 0 <= i < len(recommended_features)]
            config['data']['feature_columns'] = selected_features
        except ValueError:
            print("Invalid input, keeping all features.")
    
    # Confirm periods
    suggested_periods = config['periods']
    print(f"\nSuggested analysis periods ({len(suggested_periods)}):")
    for i, period in enumerate(suggested_periods):
        print(f"  {i+1}. {period['name']}: {period['start_date']} to {period['end_date']}")
    
    keep_periods = input(f"\nUse suggested periods? [Y/n]: ").strip().lower()
    if keep_periods in ['n', 'no']:
        print("Keeping first period only. Edit the configuration file to customize periods.")
        config['periods'] = suggested_periods[:1] if suggested_periods else []
    
    # Model selection
    models = config.get('models', {}).get('available_models', [])
    print(f"\nAvailable models: {', '.join(models)}")
    current_model = config.get('models', {}).get('default_model', 'RandomForest')
    
    new_model = input(f"Default model [{current_model}]: ").strip()
    if new_model and new_model in models:
        config['models']['default_model'] = new_model
    
    return analysis_result


def generate_config_variants(base_analysis: dict, output_dir: str):
    """Generate multiple configuration variants."""
    print(f"\nGenerating configuration variants in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DatasetAnalyzer()
    
    # Base configuration
    config_path = os.path.join(output_dir, "base_config.yaml")
    analyzer.generate_config_file(base_analysis, config_path)
    print(f"  ✅ {config_path}")
    
    # Quick test variant (single period, no augmentation)
    test_analysis = base_analysis.copy()
    test_analysis['recommended_config']['general'] = {'project_name': 'Quick_Test'}
    test_analysis['recommended_config']['periods'] = base_analysis['recommended_config']['periods'][:1]
    test_analysis['recommended_config']['models'] = {'default_model': 'KNN', 'cv_folds': 3}
    
    test_config_path = os.path.join(output_dir, "quick_test_config.yaml")
    analyzer.generate_config_file(test_analysis, test_config_path)
    print(f"  ✅ {test_config_path}")
    
    # Experiment variant (with augmentation)
    exp_analysis = base_analysis.copy()
    exp_analysis['recommended_config']['augmentation'] = {
        'enabled': True,
        'n_augment': 3,
        'noise_level': 0.05
    }
    
    exp_config_path = os.path.join(output_dir, "experiment_config.yaml")
    analyzer.generate_config_file(exp_analysis, exp_config_path)
    print(f"  ✅ {exp_config_path}")
    
    # Production variant (conservative settings)
    prod_analysis = base_analysis.copy()
    prod_analysis['recommended_config']['logging'] = {'level': 'WARNING', 'verbose': False}
    prod_analysis['recommended_config']['models'] = {
        'default_model': 'RandomForest',
        'cv_folds': 5
    }
    
    prod_config_path = os.path.join(output_dir, "production_config.yaml")
    analyzer.generate_config_file(prod_analysis, prod_config_path)
    print(f"  ✅ {prod_config_path}")


def print_analysis_summary(analysis_result: dict):
    """Print a comprehensive analysis summary."""
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Dataset info
    dataset_info = analysis_result['dataset_info']
    print(f"Dataset: {dataset_info['path']}")
    print(f"Shape: {dataset_info['shape']} (rows x columns)")
    
    # Column analysis
    columns = analysis_result['columns']
    print(f"\nColumn Analysis:")
    print(f"  Numeric columns: {len(columns['numeric_columns'])}")
    print(f"  Categorical columns: {len(columns['categorical_columns'])}")
    print(f"  DateTime columns: {len(columns['datetime_columns'])}")
    
    # Time analysis
    time_analysis = analysis_result.get('time_analysis', {})
    if time_analysis.get('recommended_time_column'):
        print(f"\nTime Analysis:")
        print(f"  Time column: {time_analysis['recommended_time_column']}")
        print(f"  Resolution: {time_analysis.get('resolution', 'unknown')}")
        if 'date_range' in time_analysis:
            date_range = time_analysis['date_range']
            print(f"  Date range: {date_range['start']} to {date_range['end']}")
            print(f"  Duration: {date_range['duration_days']} days")
    
    # Target suggestions
    target_suggestions = analysis_result['target_suggestions']
    if target_suggestions['candidates']:
        print(f"\nTarget Column Suggestions:")
        for candidate in target_suggestions['candidates'][:3]:
            print(f"  {candidate['column']} (confidence: {candidate['confidence']:.2f})")
    
    # Feature columns
    recommended_features = columns['recommended_features']
    print(f"\nRecommended Features ({len(recommended_features)}):")
    for col in recommended_features[:10]:  # Show first 10
        print(f"  - {col}")
    if len(recommended_features) > 10:
        print(f"  ... and {len(recommended_features) - 10} more")
    
    # Periods
    periods = analysis_result['recommended_config']['periods']
    print(f"\nSuggested Periods ({len(periods)}):")
    for period in periods[:5]:  # Show first 5
        print(f"  - {period['name']}: {period['start_date']} to {period['end_date']}")
    if len(periods) > 5:
        print(f"  ... and {len(periods) - 5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze datasets and generate ML pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/my_dataset.csv
  %(prog)s data/sensor_data.csv --output config/sensor_config.yaml
  %(prog)s data/emissions.csv --interactive
  %(prog)s data/wastewater.csv --generate-variants --output-dir config/variants/
        """
    )
    
    parser.add_argument(
        'data_path',
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output configuration file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for configuration files'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive configuration refinement'
    )
    
    parser.add_argument(
        '--generate-variants',
        action='store_true',
        help='Generate multiple configuration variants'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Number of rows to sample for analysis (default: 10000)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze only, do not generate configuration files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Dataset file not found: {args.data_path}")
        return 1
    
    try:
        # Analyze dataset
        print(f"🔍 Analyzing dataset: {args.data_path}")
        analyzer = DatasetAnalyzer()
        analysis_result = analyzer.analyze_dataset(args.data_path, args.sample_size)
        
        # Print analysis summary
        print_analysis_summary(analysis_result)
        
        # Interactive refinement if requested
        if args.interactive:
            analysis_result = interactive_configuration(analysis_result)
        
        # Generate configurations if not dry run
        if not args.dry_run:
            if args.generate_variants:
                # Generate multiple variants
                output_dir = args.output_dir or 'config/auto_generated'
                generate_config_variants(analysis_result, output_dir)
                
            else:
                # Generate single configuration
                output_path = args.output
                if not output_path:
                    dataset_name = Path(args.data_path).stem
                    output_path = f"config/auto_{dataset_name}.yaml"
                
                analyzer.generate_config_file(analysis_result, output_path)
                print(f"\n✅ Configuration generated: {output_path}")
                
                # Quick test command
                print(f"\n🚀 Test the configuration:")
                print(f"python main.py --config {output_path} --dry-run")
        
        print(f"\n🎉 Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())