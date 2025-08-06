#!/usr/bin/env python3
"""
BetterAI Command Line Interface

A unified CLI for the BetterAI reward model training and evaluation framework.
Provides easy access to all features including training, validation, evaluation, and analysis.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from train_reward_model import RewardModelTrainer, load_config_from_file, create_default_config
    from data_validator import DataValidator
    from evaluation_metrics import RewardModelEvaluator
    from human_evaluation import HumanEvaluationFramework
    from model_comparison import ModelComparisonTool
    from evaluate_reward_model import RewardModelEvaluatorApp
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BetterAICLI:
    """Main CLI class for BetterAI framework."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def validate_data(self, args):
        """Validate and clean data."""
        if not MODULES_AVAILABLE:
            logger.error("Data validation module not available")
            return
        
        logger.info("🔍 Starting data validation...")
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "min_length": args.min_length,
                "max_length": args.max_length,
                "max_similarity": args.max_similarity,
                "min_quality_score": args.min_quality_score,
                "remove_html": args.remove_html,
                "remove_urls": args.remove_urls
            }
        
        # Create validator
        validator = DataValidator(config)
        
        # Load and validate data
        df = pd.read_csv(args.input)
        cleaned_df, results = validator.validate_dataset(df)
        
        # Save results
        cleaned_df.to_csv(args.output, index=False)
        
        if args.report:
            validator.save_validation_report(args.report)
        else:
            report_path = Path(args.output).parent / "validation_report.json"
            validator.save_validation_report(str(report_path))
        
        # Print summary
        validator.print_summary()
        
        logger.info(f"✅ Data validation completed. Cleaned data saved to: {args.output}")
    
    def train_model(self, args):
        """Train reward model."""
        if not MODULES_AVAILABLE:
            logger.error("Training module not available")
            return
        
        logger.info("🏋️ Starting model training...")
        
        # Load configuration
        if args.config:
            config = load_config_from_file(args.config)
        else:
            config = create_default_config()
        
        # Override config with command line arguments
        if args.model_name:
            config["model_name"] = args.model_name
        if args.data_path:
            config["data_path"] = args.data_path
        if args.output_dir:
            config["output_dir"] = args.output_dir
        if args.num_epochs:
            config["num_epochs"] = args.num_epochs
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        
        # Create trainer
        trainer = RewardModelTrainer(config)
        
        # Train model
        trainer.load_tokenizer_and_model()
        datasets = trainer.load_and_preprocess_data()
        trainer.create_trainer(datasets["train"], datasets["test"])
        train_result = trainer.train()
        
        logger.info(f"✅ Training completed. Model saved to: {trainer.output_dir}")
    
    def evaluate_model(self, args):
        """Evaluate reward model."""
        if not MODULES_AVAILABLE:
            logger.error("Evaluation module not available")
            return
        
        logger.info("🔍 Starting model evaluation...")
        
        # Create evaluator app
        config = {
            "output_dir": args.output_dir,
            "human_evaluation": {
                "evaluation_questions": [
                    "Which response is more helpful?",
                    "Which response is more accurate?",
                    "Which response is more appropriate?"
                ],
                "evaluation_scale": 5,
                "min_evaluations_per_sample": 3
            }
        }
        
        evaluator_app = RewardModelEvaluatorApp(config)
        
        # Load model and data
        evaluator_app.load_model(args.model_path)
        test_data = evaluator_app.load_test_data(args.test_data)
        
        # Run evaluation
        results = evaluator_app.run_comprehensive_evaluation(test_data)
        
        # Handle human evaluation
        if args.human_eval:
            human_evaluator, interface = evaluator_app.run_human_evaluation_setup(test_data)
            if interface:
                logger.info("🖥️ Starting human evaluation interface...")
                interface.launch(share=False)
        
        # Handle model comparison
        if args.compare_models:
            other_models = {}
            for model_pair in args.compare_models.split(','):
                if ':' in model_pair:
                    name, path = model_pair.split(':', 1)
                    other_models[name.strip()] = path.strip()
            
            if other_models:
                comparison_results = evaluator_app.run_model_comparison(other_models, test_data)
        
        # Print summary
        evaluator_app.print_summary()
        
        logger.info(f"✅ Evaluation completed. Results saved to: {evaluator_app.output_dir}")
    
    def compare_models(self, args):
        """Compare multiple models."""
        if not MODULES_AVAILABLE:
            logger.error("Model comparison module not available")
            return
        
        logger.info("⚖️ Starting model comparison...")
        
        # Load models and tokenizers
        models = {}
        tokenizers = {}
        
        for model_pair in args.models.split(','):
            if ':' in model_pair:
                name, path = model_pair.split(':', 1)
                name = name.strip()
                path = path.strip()
                
                try:
                    logger.info(f"📋 Loading model: {name}")
                    tokenizer = AutoTokenizer.from_pretrained(path)
                    model = AutoModelForSequenceClassification.from_pretrained(path)
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    models[name] = model
                    tokenizers[name] = tokenizer
                    
                except Exception as e:
                    logger.error(f"❌ Error loading model {name}: {str(e)}")
                    continue
        
        if len(models) < 2:
            logger.error("Need at least 2 models for comparison")
            return
        
        # Load test data
        test_data = pd.read_csv(args.test_data)
        
        # Create comparison tool
        comparison_tool = ModelComparisonTool(models, tokenizers, self.device)
        
        # Run comparison
        results = comparison_tool.compare_models_on_dataset(test_data)
        
        # Generate report
        comparison_tool.create_comparison_report(args.output)
        
        # Create visualizations
        if args.visualize:
            output_dir = Path(args.output).parent
            comparison_tool.create_comparison_visualizations(str(output_dir))
        
        logger.info(f"✅ Model comparison completed. Report saved to: {args.output}")
    
    def launch_demo(self, args):
        """Launch web demo."""
        logger.info("🎮 Launching web demo...")
        
        try:
            import subprocess
            subprocess.run([sys.executable, "app.py"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error launching demo: {e}")
        except FileNotFoundError:
            logger.error("❌ app.py not found")
    
    def create_config(self, args):
        """Create configuration files."""
        logger.info("⚙️ Creating configuration files...")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        training_config = {
            "model_name": "roberta-base",
            "data_path": "data/pairwise_preferences.csv",
            "output_dir": "./models/reward_model",
            "max_length": 512,
            "num_epochs": 3,
            "batch_size": 4,
            "eval_batch_size": 4,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 3,
            "validation_split": 0.1,
            "use_wandb": False,
            "use_fp16": False,
            "use_bf16": False,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "seed": 42,
            "data_validation": {
                "enabled": True,
                "min_length": 10,
                "max_length": 2000,
                "min_chosen_length": 10,
                "max_chosen_length": 2000,
                "min_rejected_length": 10,
                "max_rejected_length": 2000,
                "max_similarity": 0.95,
                "min_quality_score": 0.0,
                "remove_html": True,
                "remove_urls": False,
                "save_cleaned_data": True,
                "save_validation_report": True
            }
        }
        
        # Validation config
        validation_config = {
            "min_length": 10,
            "max_length": 2000,
            "min_chosen_length": 20,
            "max_chosen_length": 1500,
            "min_rejected_length": 20,
            "max_rejected_length": 1500,
            "max_similarity": 0.90,
            "min_quality_score": 0.3,
            "remove_html": True,
            "remove_urls": False,
            "save_cleaned_data": True,
            "save_validation_report": True
        }
        
        # Evaluation config
        evaluation_config = {
            "output_dir": "./evaluation_results",
            "human_evaluation": {
                "evaluation_questions": [
                    "Which response is more helpful?",
                    "Which response is more accurate?",
                    "Which response is more appropriate?"
                ],
                "evaluation_scale": 5,
                "min_evaluations_per_sample": 3
            },
            "bias_indicators": {
                "gender": ["he", "she", "his", "her", "him", "man", "woman"],
                "race": ["black", "white", "asian", "hispanic", "african"],
                "age": ["young", "old", "elderly", "teenager", "senior"]
            }
        }
        
        # Save configs
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        
        with open(output_dir / "validation_config.json", 'w') as f:
            json.dump(validation_config, f, indent=2)
        
        with open(output_dir / "evaluation_config.json", 'w') as f:
            json.dump(evaluation_config, f, indent=2)
        
        logger.info(f"✅ Configuration files created in: {output_dir}")
        logger.info("   - training_config.json")
        logger.info("   - validation_config.json")
        logger.info("   - evaluation_config.json")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="BetterAI: Advanced Reward Model Training & Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate data
  betterai validate --input data.csv --output cleaned_data.csv
  
  # Train model
  betterai train --data-path data.csv --output-dir ./models
  
  # Evaluate model
  betterai evaluate --model-path ./models/reward_model --test-data test.csv
  
  # Compare models
  betterai compare --models model1:./models/model1,model2:./models/model2 --test-data test.csv
  
  # Launch demo
  betterai demo
  
  # Create configs
  betterai config --output-dir ./configs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate and clean data')
    validate_parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    validate_parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    validate_parser.add_argument('--config', '-c', help='Configuration file')
    validate_parser.add_argument('--report', '-r', help='Validation report file')
    validate_parser.add_argument('--min-length', type=int, default=10, help='Minimum text length')
    validate_parser.add_argument('--max-length', type=int, default=2000, help='Maximum text length')
    validate_parser.add_argument('--max-similarity', type=float, default=0.95, help='Maximum similarity')
    validate_parser.add_argument('--min-quality-score', type=float, default=0.0, help='Minimum quality score')
    validate_parser.add_argument('--remove-html', action='store_true', help='Remove HTML content')
    validate_parser.add_argument('--remove-urls', action='store_true', help='Remove URL content')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train reward model')
    train_parser.add_argument('--config', '-c', help='Configuration file')
    train_parser.add_argument('--model-name', help='Model name')
    train_parser.add_argument('--data-path', help='Data path')
    train_parser.add_argument('--output-dir', help='Output directory')
    train_parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate reward model')
    evaluate_parser.add_argument('--model-path', '-m', required=True, help='Model path')
    evaluate_parser.add_argument('--test-data', '-t', required=True, help='Test data file')
    evaluate_parser.add_argument('--output-dir', '-o', default='./evaluation_results', help='Output directory')
    evaluate_parser.add_argument('--human-eval', action='store_true', help='Enable human evaluation')
    evaluate_parser.add_argument('--compare-models', help='Models to compare (model1:path1,model2:path2)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', '-m', required=True, help='Models to compare (model1:path1,model2:path2)')
    compare_parser.add_argument('--test-data', '-t', required=True, help='Test data file')
    compare_parser.add_argument('--output', '-o', default='comparison_report.md', help='Output report file')
    compare_parser.add_argument('--visualize', '-v', action='store_true', help='Create visualizations')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Launch web demo')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Create configuration files')
    config_parser.add_argument('--output-dir', '-o', default='./config', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = BetterAICLI()
    
    try:
        # Execute command
        if args.command == 'validate':
            cli.validate_data(args)
        elif args.command == 'train':
            cli.train_model(args)
        elif args.command == 'evaluate':
            cli.evaluate_model(args)
        elif args.command == 'compare':
            cli.compare_models(args)
        elif args.command == 'demo':
            cli.launch_demo(args)
        elif args.command == 'config':
            cli.create_config(args)
        
    except Exception as e:
        logger.error(f"❌ Command failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 