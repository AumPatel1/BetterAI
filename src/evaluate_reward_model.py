#!/usr/bin/env python3
"""
Comprehensive reward model evaluation script.

This script provides a complete evaluation framework for reward models including:
- Custom evaluation metrics
- Bias detection and analysis
- Human evaluation framework
- Model comparison tools
- Evaluation reports and visualizations

Usage:
    python evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import sys
import os
from typing import Dict, Any, Optional

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from evaluation_metrics import RewardModelEvaluator
    from human_evaluation import HumanEvaluationFramework
    from model_comparison import ModelComparisonTool
    EVALUATION_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some evaluation modules not available: {e}")
    EVALUATION_MODULES_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RewardModelEvaluatorApp:
    """Main application for comprehensive reward model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.evaluator = None
        self.results = {}
        
        # Create output directory
        self.output_dir = Path(config.get("output_dir", "./evaluation_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Initializing RewardModelEvaluatorApp on device: {self.device}")
    
    def load_model(self, model_path: str):
        """Load the reward model and tokenizer."""
        try:
            logger.info(f"📋 Loading model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize evaluator
            if EVALUATION_MODULES_AVAILABLE:
                self.evaluator = RewardModelEvaluator(self.model, self.tokenizer, self.device)
            
            logger.info(f"✅ Model loaded successfully with {self.model.num_parameters():,} parameters")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def load_test_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate test data."""
        try:
            logger.info(f"📊 Loading test data from: {data_path}")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Test data file not found: {data_path}")
            
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            # Validate required columns
            required_columns = ["chosen", "rejected"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"✅ Test data loaded: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading test data: {str(e)}")
            raise
    
    def run_comprehensive_evaluation(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive evaluation including all metrics and analyses."""
        logger.info("🔍 Running comprehensive evaluation...")
        
        if not EVALUATION_MODULES_AVAILABLE or self.evaluator is None:
            logger.error("Evaluation modules not available")
            return {}
        
        # Run all evaluations
        evaluation_results = {}
        
        # 1. Preference accuracy
        logger.info("🎯 Evaluating preference accuracy...")
        evaluation_results["preference_accuracy"] = self.evaluator.evaluate_preference_accuracy(test_data)
        
        # 2. Ranking metrics
        logger.info("📊 Evaluating ranking metrics...")
        evaluation_results["ranking_metrics"] = self.evaluator.evaluate_ranking_metrics(test_data)
        
        # 3. Bias detection
        logger.info("🔍 Running bias detection...")
        evaluation_results["bias_detection"] = self.evaluator.evaluate_bias_detection(test_data)
        
        # 4. Quality distribution
        logger.info("📈 Analyzing quality distribution...")
        evaluation_results["quality_distribution"] = self.evaluator.evaluate_response_quality_distribution(test_data)
        
        # 5. Model consistency
        logger.info("🔄 Evaluating model consistency...")
        evaluation_results["model_consistency"] = self.evaluator.evaluate_model_consistency(test_data)
        
        # 6. Generate comprehensive report
        logger.info("📋 Generating comprehensive report...")
        comprehensive_report = self.evaluator.generate_evaluation_report(
            test_data, 
            str(self.output_dir / "evaluation_report.json")
        )
        evaluation_results["comprehensive_report"] = comprehensive_report
        
        # 7. Create visualizations
        logger.info("📊 Creating visualizations...")
        self.evaluator.create_visualizations(test_data, str(self.output_dir))
        
        self.results = evaluation_results
        return evaluation_results
    
    def run_human_evaluation_setup(self, test_data: pd.DataFrame, num_samples: int = 50):
        """Set up human evaluation framework."""
        if not EVALUATION_MODULES_AVAILABLE:
            logger.error("Human evaluation module not available")
            return None
        
        logger.info("👥 Setting up human evaluation framework...")
        
        # Create human evaluation framework
        human_eval_config = self.config.get("human_evaluation", {})
        human_evaluator = HumanEvaluationFramework(human_eval_config)
        
        # Generate model predictions for human evaluation
        model_outputs = []
        for i, row in test_data.head(num_samples).iterrows():
            chosen_score = self._get_reward_score(row['chosen'])
            rejected_score = self._get_reward_score(row['rejected'])
            
            output = {
                "prompt": f"Sample {i}",
                "response_a": row['chosen'],
                "response_b": row['rejected'],
                "model_preference": "A" if chosen_score > rejected_score else "B",
                "model_confidence": abs(chosen_score - rejected_score)
            }
            model_outputs.append(output)
        
        # Create evaluation dataset
        evaluation_dataset = human_evaluator.create_evaluation_dataset(model_outputs, num_samples)
        
        # Save evaluation dataset
        evaluation_dataset.to_csv(self.output_dir / "human_evaluation_dataset.csv", index=False)
        logger.info(f"💾 Human evaluation dataset saved to: {self.output_dir / 'human_evaluation_dataset.csv'}")
        
        # Create Gradio interface
        interface = human_evaluator.create_gradio_interface(evaluation_dataset)
        
        return human_evaluator, interface
    
    def run_model_comparison(self, other_models: Dict[str, str], test_data: pd.DataFrame):
        """Run model comparison with other models."""
        if not EVALUATION_MODULES_AVAILABLE:
            logger.error("Model comparison module not available")
            return None
        
        logger.info("⚖️ Running model comparison...")
        
        # Load other models
        models = {"current": self.model}
        tokenizers = {"current": self.tokenizer}
        
        for model_name, model_path in other_models.items():
            try:
                logger.info(f"📋 Loading comparison model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                models[model_name] = model
                tokenizers[model_name] = tokenizer
                
            except Exception as e:
                logger.error(f"❌ Error loading model {model_name}: {str(e)}")
                continue
        
        # Create comparison tool
        comparison_tool = ModelComparisonTool(models, tokenizers, self.device)
        
        # Run comparison
        comparison_results = comparison_tool.compare_models_on_dataset(test_data)
        
        # Generate comparison report
        comparison_report = comparison_tool.create_comparison_report(
            str(self.output_dir / "model_comparison_report.md")
        )
        
        # Create comparison visualizations
        comparison_tool.create_comparison_visualizations(str(self.output_dir))
        
        return comparison_results
    
    def _get_reward_score(self, text: str) -> float:
        """Get reward score for a given text."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            score = outputs.logits.item()
            return score
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("📊 REWARD MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Preference accuracy
        if "preference_accuracy" in self.results:
            acc = self.results["preference_accuracy"]["preference_accuracy"]
            print(f"Preference Accuracy: {acc:.4f}")
        
        # Ranking metrics
        if "ranking_metrics" in self.results:
            rm = self.results["ranking_metrics"]
            print(f"NDCG: {rm['ndcg']:.4f}")
            print(f"MRR: {rm['mrr']:.4f}")
            print(f"Kendall's Tau: {rm['kendall_tau']:.4f}")
        
        # Model consistency
        if "model_consistency" in self.results:
            cons = self.results["model_consistency"]["consistency_score"]
            print(f"Model Consistency: {cons:.4f}")
        
        # Overall score
        if "comprehensive_report" in self.results:
            overall = self.results["comprehensive_report"].get("overall_score", 0)
            print(f"Overall Score: {overall:.4f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)

def create_default_config() -> Dict[str, Any]:
    """Create default evaluation configuration."""
    return {
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

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive reward model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv
  
  # Evaluation with custom output directory
  python evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --output-dir ./my_results
  
  # Evaluation with human evaluation setup
  python evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --human-eval
  
  # Model comparison
  python evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --compare-models model1:./models/model1,model2:./models/model2
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to the reward model directory"
    )
    
    parser.add_argument(
        "--test-data", "-t",
        required=True,
        help="Path to test data file (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--human-eval",
        action="store_true",
        help="Set up human evaluation framework"
    )
    
    parser.add_argument(
        "--compare-models",
        help="Comma-separated list of model_name:model_path pairs for comparison"
    )
    
    parser.add_argument(
        "--config",
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Update config with command line arguments
    config["output_dir"] = args.output_dir
    
    try:
        # Initialize evaluator
        evaluator_app = RewardModelEvaluatorApp(config)
        
        # Load model
        evaluator_app.load_model(args.model_path)
        
        # Load test data
        test_data = evaluator_app.load_test_data(args.test_data)
        
        # Run comprehensive evaluation
        results = evaluator_app.run_comprehensive_evaluation(test_data)
        
        # Set up human evaluation if requested
        if args.human_eval:
            human_evaluator, interface = evaluator_app.run_human_evaluation_setup(test_data)
            if interface:
                logger.info("🖥️ Starting human evaluation interface...")
                interface.launch(share=False)
        
        # Run model comparison if requested
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
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 