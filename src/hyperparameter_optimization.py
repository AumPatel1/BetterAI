"""
Hyperparameter optimization for reward models.

This module provides automated hyperparameter tuning using Optuna,
with support for various search strategies and optimization objectives.
"""

import optuna
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Import project modules
from train_reward_model import RewardModelTrainer, load_config_from_file, create_default_config
from evaluation_metrics import RewardModelEvaluator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, base_config: Dict[str, Any], optimization_config: Dict[str, Any]):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            base_config: Base configuration for training
            optimization_config: Configuration for optimization process
        """
        self.base_config = base_config
        self.optimization_config = optimization_config
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trials_history = []
        
        # Create output directory
        self.output_dir = Path(optimization_config.get("output_dir", "./hyperparameter_optimization"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔍 Initialized hyperparameter optimizer with {self.device} device")
    
    def create_objective_function(self, train_data_path: str, val_data_path: str) -> Callable:
        """
        Create the objective function for optimization.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            """Objective function for hyperparameter optimization."""
            try:
                # Suggest hyperparameters
                params = self._suggest_hyperparameters(trial)
                
                # Update base config with suggested parameters
                config = self.base_config.copy()
                config.update(params)
                
                # Set trial-specific output directory
                trial_dir = self.output_dir / f"trial_{trial.number}"
                config["output_dir"] = str(trial_dir)
                
                # Create trainer
                trainer = RewardModelTrainer(config)
                
                # Load data
                trainer.load_tokenizer_and_model()
                datasets = trainer.load_and_preprocess_data()
                
                # Create trainer with validation data
                trainer.create_trainer(datasets["train"], datasets["test"])
                
                # Train model
                train_result = trainer.train()
                
                # Evaluate model
                evaluator = RewardModelEvaluator(
                    trainer.model, 
                    trainer.tokenizer, 
                    device=self.device
                )
                
                # Load validation data for evaluation
                val_data = pd.read_csv(val_data_path)
                
                # Run evaluation
                evaluation_results = evaluator.evaluate_preference_accuracy(val_data)
                
                # Calculate objective score (higher is better)
                objective_score = evaluation_results.get("accuracy", 0.0)
                
                # Add additional metrics to trial
                trial.set_user_attr("precision", evaluation_results.get("precision", 0.0))
                trial.set_user_attr("recall", evaluation_results.get("recall", 0.0))
                trial.set_user_attr("f1_score", evaluation_results.get("f1_score", 0.0))
                trial.set_user_attr("training_loss", train_result.metrics.get("train_loss", float('inf')))
                
                logger.info(f"Trial {trial.number}: Score = {objective_score:.4f}")
                
                return objective_score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf')  # Return worst possible score
        
        return objective
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        params = {}
        
        # Learning rate
        if self.optimization_config.get("optimize_learning_rate", True):
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 
                1e-6, 1e-4, 
                log=True
            )
        
        # Batch size
        if self.optimization_config.get("optimize_batch_size", True):
            params["batch_size"] = trial.suggest_categorical(
                "batch_size", 
                [2, 4, 8, 16, 32]
            )
        
        # Number of epochs
        if self.optimization_config.get("optimize_num_epochs", True):
            params["num_epochs"] = trial.suggest_int("num_epochs", 1, 10)
        
        # Weight decay
        if self.optimization_config.get("optimize_weight_decay", True):
            params["weight_decay"] = trial.suggest_float(
                "weight_decay", 
                0.0, 0.1
            )
        
        # Warmup steps
        if self.optimization_config.get("optimize_warmup_steps", True):
            params["warmup_steps"] = trial.suggest_int("warmup_steps", 0, 1000)
        
        # Gradient accumulation steps
        if self.optimization_config.get("optimize_gradient_accumulation", True):
            params["gradient_accumulation_steps"] = trial.suggest_int(
                "gradient_accumulation_steps", 
                1, 8
            )
        
        # Model architecture parameters
        if self.optimization_config.get("optimize_model_architecture", True):
            # Dropout rate
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
            
            # Hidden layer sizes for custom architectures
            if "model_type" in self.base_config and self.base_config["model_type"] == "multilayer":
                hidden_size = self.base_config.get("hidden_size", 768)
                layer_sizes = [
                    hidden_size,
                    trial.suggest_int("hidden_1", hidden_size // 4, hidden_size // 2),
                    trial.suggest_int("hidden_2", hidden_size // 8, hidden_size // 4),
                    1
                ]
                params["layer_sizes"] = layer_sizes
        
        # Data validation parameters
        if self.optimization_config.get("optimize_data_validation", True):
            params["data_validation"] = {
                "min_length": trial.suggest_int("min_length", 5, 50),
                "max_length": trial.suggest_int("max_length", 1000, 3000),
                "max_similarity": trial.suggest_float("max_similarity", 0.8, 0.99)
            }
        
        return params
    
    def optimize(self, train_data_path: str, val_data_path: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"🚀 Starting hyperparameter optimization with {n_trials} trials")
        
        # Create study
        study_name = f"reward_model_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage = self.optimization_config.get("storage", None)
        
        if storage:
            self.study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(direction="maximize", study_name=study_name)
        
        # Create objective function
        objective = self.create_objective_function(train_data_path, val_data_path)
        
        # Run optimization
        start_time = time.time()
        self.study.optimize(objective, n_trials=n_trials)
        end_time = time.time()
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Save results
        self._save_optimization_results()
        
        logger.info(f"✅ Optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_score": self.best_score,
            "best_params": self.best_params,
            "study": self.study,
            "optimization_time": end_time - start_time
        }
    
    def _save_optimization_results(self):
        """Save optimization results to files."""
        # Save best parameters
        best_params_path = self.output_dir / "best_parameters.json"
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save study results
        study_path = self.output_dir / "optimization_study.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(self.study, f)
        
        # Create optimization report
        self._create_optimization_report()
        
        logger.info(f"💾 Optimization results saved to {self.output_dir}")
    
    def _create_optimization_report(self):
        """Create a comprehensive optimization report."""
        report_path = self.output_dir / "optimization_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Best Score:** {self.best_score:.4f}\n\n")
            f.write(f"**Total Trials:** {len(self.study.trials)}\n\n")
            
            f.write("## Best Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.best_params, indent=2))
            f.write("\n```\n\n")
            
            f.write("## Parameter Importance\n\n")
            try:
                importance = optuna.importance.get_param_importances(self.study)
                for param, imp in importance.items():
                    f.write(f"- **{param}:** {imp:.4f}\n")
            except Exception as e:
                f.write(f"Could not calculate parameter importance: {e}\n")
            
            f.write("\n## Trial History\n\n")
            f.write("| Trial | Score | Learning Rate | Batch Size | Epochs |\n")
            f.write("|-------|-------|---------------|------------|--------|\n")
            
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    score = trial.value
                    lr = trial.params.get("learning_rate", "N/A")
                    batch_size = trial.params.get("batch_size", "N/A")
                    epochs = trial.params.get("num_epochs", "N/A")
                    f.write(f"| {trial.number} | {score:.4f} | {lr} | {batch_size} | {epochs} |\n")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration with optimized parameters."""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        config = self.base_config.copy()
        config.update(self.best_params)
        return config
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
            axes[0, 0].set_title("Optimization History")
            
            # Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
            axes[0, 1].set_title("Parameter Importance")
            
            # Parameter relationships
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
            axes[1, 0].set_title("Parameter Relationships")
            
            # Parameter contour
            if len(self.best_params) >= 2:
                param_names = list(self.best_params.keys())[:2]
                optuna.visualization.matplotlib.plot_contour(self.study, params=param_names, ax=axes[1, 1])
                axes[1, 1].set_title(f"Contour Plot: {param_names[0]} vs {param_names[1]}")
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / "optimization_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Optimization plots saved to {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

def create_optimization_config() -> Dict[str, Any]:
    """Create a default optimization configuration."""
    return {
        "output_dir": "./hyperparameter_optimization",
        "storage": None,  # SQLite database path for persistent storage
        "optimize_learning_rate": True,
        "optimize_batch_size": True,
        "optimize_num_epochs": True,
        "optimize_weight_decay": True,
        "optimize_warmup_steps": True,
        "optimize_gradient_accumulation": True,
        "optimize_model_architecture": True,
        "optimize_data_validation": True,
        "n_trials": 50,
        "timeout": 3600,  # 1 hour timeout
        "early_stopping_patience": 10
    }

if __name__ == "__main__":
    # Example usage
    base_config = create_default_config()
    optimization_config = create_optimization_config()
    
    optimizer = HyperparameterOptimizer(base_config, optimization_config)
    
    # Run optimization
    results = optimizer.optimize(
        train_data_path="data/train.csv",
        val_data_path="data/val.csv",
        n_trials=20
    )
    
    # Get best configuration
    best_config = optimizer.get_best_config()
    print(f"Best configuration: {best_config}") 