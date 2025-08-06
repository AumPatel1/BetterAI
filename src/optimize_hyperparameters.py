#!/usr/bin/env python3
"""
Hyperparameter optimization CLI script.

This script provides a command-line interface for running hyperparameter
optimization on reward models using Optuna.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from hyperparameter_optimization import HyperparameterOptimizer, create_optimization_config
from train_reward_model import load_config_from_file, create_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for hyperparameter optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for reward models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python optimize_hyperparameters.py --train-data data/train.csv --val-data data/val.csv
  
  # Custom optimization config
  python optimize_hyperparameters.py --train-data data/train.csv --val-data data/val.csv --optimization-config config/optimization.json
  
  # Limited trials
  python optimize_hyperparameters.py --train-data data/train.csv --val-data data/val.csv --n-trials 20
  
  # Save best config
  python optimize_hyperparameters.py --train-data data/train.csv --val-data data/val.csv --save-best-config best_config.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--train-data", 
        required=True, 
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--val-data", 
        required=True, 
        help="Path to validation data CSV file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--base-config", 
        help="Path to base training configuration JSON file"
    )
    parser.add_argument(
        "--optimization-config", 
        help="Path to optimization configuration JSON file"
    )
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=50, 
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./hyperparameter_optimization", 
        help="Output directory for optimization results (default: ./hyperparameter_optimization)"
    )
    parser.add_argument(
        "--save-best-config", 
        help="Path to save the best configuration found"
    )
    parser.add_argument(
        "--storage", 
        help="Optuna storage URL for persistent optimization (e.g., sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        help="Optimization timeout in seconds"
    )
    parser.add_argument(
        "--plot", 
        action="store_true", 
        help="Generate optimization plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        if args.base_config:
            base_config = load_config_from_file(args.base_config)
        else:
            base_config = create_default_config()
            logger.info("Using default base configuration")
        
        if args.optimization_config:
            with open(args.optimization_config, 'r') as f:
                optimization_config = json.load(f)
        else:
            optimization_config = create_optimization_config()
            logger.info("Using default optimization configuration")
        
        # Override optimization config with command line arguments
        optimization_config["output_dir"] = args.output_dir
        optimization_config["n_trials"] = args.n_trials
        
        if args.storage:
            optimization_config["storage"] = args.storage
        
        if args.timeout:
            optimization_config["timeout"] = args.timeout
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(base_config, optimization_config)
        
        # Run optimization
        logger.info(f"🚀 Starting hyperparameter optimization with {args.n_trials} trials")
        logger.info(f"📁 Output directory: {args.output_dir}")
        
        results = optimizer.optimize(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            n_trials=args.n_trials
        )
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best Score: {results['best_score']:.4f}")
        print(f"Optimization Time: {results['optimization_time']:.2f} seconds")
        print(f"Best Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        # Save best configuration
        if args.save_best_config:
            best_config = optimizer.get_best_config()
            with open(args.save_best_config, 'w') as f:
                json.dump(best_config, f, indent=2)
            logger.info(f"💾 Best configuration saved to: {args.save_best_config}")
        
        # Generate plots
        if args.plot:
            try:
                optimizer.plot_optimization_history()
                logger.info("📊 Optimization plots generated")
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        
        # Print summary
        print(f"\n📊 Optimization completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📄 Report: {args.output_dir}/optimization_report.md")
        
        if args.plot:
            print(f"📈 Plots: {args.output_dir}/optimization_plots.png")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 