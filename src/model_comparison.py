import torch
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelComparisonTool:
    """Comprehensive tool for comparing multiple reward models."""
    
    def __init__(self, models: Dict[str, Any], tokenizers: Dict[str, Any], device: str = "cpu"):
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        self.comparison_results = {}
        
        # Validate inputs
        if not isinstance(models, dict) or not isinstance(tokenizers, dict):
            raise ValueError("Models and tokenizers must be dictionaries")
        
        if set(models.keys()) != set(tokenizers.keys()):
            raise ValueError("Model and tokenizer keys must match")
        
        # Move models to device and set to eval mode
        for name, model in self.models.items():
            model.to(device)
            model.eval()
        
        logger.info(f"🔧 Initialized comparison tool with {len(models)} models")
    
    def compare_models_on_dataset(self, test_data: pd.DataFrame, 
                                 metrics: List[str] = None) -> Dict[str, Any]:
        """Compare all models on the same test dataset."""
        logger.info("⚖️ Comparing models on test dataset...")
        
        if metrics is None:
            metrics = ["preference_accuracy", "ranking_metrics", "bias_detection", "consistency"]
        
        results = {}
        
        for model_name in self.models.keys():
            logger.info(f"📊 Evaluating {model_name}...")
            model_results = self._evaluate_single_model(model_name, test_data, metrics)
            results[model_name] = model_results
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(results, test_data)
        
        self.comparison_results = {
            "individual_results": results,
            "comparison_metrics": comparison_metrics,
            "model_names": list(self.models.keys()),
            "test_data_size": len(test_data)
        }
        
        logger.info("✅ Model comparison completed")
        return self.comparison_results
    
    def _evaluate_single_model(self, model_name: str, test_data: pd.DataFrame, 
                              metrics: List[str]) -> Dict[str, Any]:
        """Evaluate a single model on the test dataset."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        results = {}
        
        # Calculate reward scores for all samples
        chosen_scores = []
        rejected_scores = []
        
        for _, row in test_data.iterrows():
            chosen_score = self._get_reward_score(model, tokenizer, row['chosen'])
            rejected_score = self._get_reward_score(model, tokenizer, row['rejected'])
            
            chosen_scores.append(chosen_score)
            rejected_scores.append(rejected_score)
        
        # Preference accuracy
        if "preference_accuracy" in metrics:
            correct_predictions = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
            accuracy = correct_predictions / len(test_data)
            results["preference_accuracy"] = accuracy
        
        # Ranking metrics
        if "ranking_metrics" in metrics:
            ndcg = self._calculate_ndcg(chosen_scores, rejected_scores)
            mrr = self._calculate_mrr(chosen_scores, rejected_scores)
            kendall_tau = self._calculate_kendall_tau(chosen_scores, rejected_scores)
            
            results["ranking_metrics"] = {
                "ndcg": ndcg,
                "mrr": mrr,
                "kendall_tau": kendall_tau
            }
        
        # Bias detection
        if "bias_detection" in metrics:
            bias_results = self._detect_bias(test_data, chosen_scores, rejected_scores)
            results["bias_detection"] = bias_results
        
        # Consistency
        if "consistency" in metrics:
            consistency = self._calculate_consistency(model, tokenizer, test_data)
            results["consistency"] = consistency
        
        # Score statistics
        results["score_statistics"] = {
            "chosen_mean": np.mean(chosen_scores),
            "chosen_std": np.std(chosen_scores),
            "rejected_mean": np.mean(rejected_scores),
            "rejected_std": np.std(rejected_scores),
            "score_difference_mean": np.mean([c - r for c, r in zip(chosen_scores, rejected_scores)])
        }
        
        return results
    
    def _calculate_comparison_metrics(self, results: Dict[str, Any], 
                                    test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for comparing models."""
        model_names = list(results.keys())
        comparison_metrics = {}
        
        # Agreement between models
        agreement_matrix = self._calculate_agreement_matrix(model_names, test_data)
        comparison_metrics["agreement_matrix"] = agreement_matrix
        
        # Performance ranking
        if "preference_accuracy" in results[model_names[0]]:
            accuracies = [results[name]["preference_accuracy"] for name in model_names]
            ranking = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
            comparison_metrics["performance_ranking"] = ranking
        
        # Statistical significance
        if len(model_names) >= 2:
            significance_tests = self._calculate_significance_tests(results, test_data)
            comparison_metrics["significance_tests"] = significance_tests
        
        return comparison_metrics
    
    def _calculate_agreement_matrix(self, model_names: List[str], 
                                  test_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate agreement matrix between all model pairs."""
        n_models = len(model_names)
        agreement_matrix = pd.DataFrame(index=model_names, columns=model_names)
        
        # Get predictions for all models
        predictions = {}
        for model_name in model_names:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            model_predictions = []
            for _, row in test_data.iterrows():
                chosen_score = self._get_reward_score(model, tokenizer, row['chosen'])
                rejected_score = self._get_reward_score(model, tokenizer, row['rejected'])
                model_predictions.append(chosen_score > rejected_score)
            
            predictions[model_name] = model_predictions
        
        # Calculate agreement between each pair
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix.loc[model1, model2] = 1.0
                else:
                    pred1 = predictions[model1]
                    pred2 = predictions[model2]
                    agreement = sum(p1 == p2 for p1, p2 in zip(pred1, pred2)) / len(pred1)
                    agreement_matrix.loc[model1, model2] = agreement
        
        return agreement_matrix
    
    def _calculate_significance_tests(self, results: Dict[str, Any], 
                                    test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance tests between models."""
        model_names = list(results.keys())
        significance_results = {}
        
        if len(model_names) < 2:
            return significance_results
        
        # McNemar's test for paired comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:
                    continue
                
                # Get predictions
                pred1 = self._get_model_predictions(model1, test_data)
                pred2 = self._get_model_predictions(model2, test_data)
                
                # Calculate McNemar's test statistic
                # This is a simplified version - in practice, you'd use scipy.stats.mcnemar
                a = sum(p1 and p2 for p1, p2 in zip(pred1, pred2))  # Both correct
                b = sum(not p1 and p2 for p1, p2 in zip(pred1, pred2))  # Model2 correct, Model1 wrong
                c = sum(p1 and not p2 for p1, p2 in zip(pred1, pred2))  # Model1 correct, Model2 wrong
                d = sum(not p1 and not p2 for p1, p2 in zip(pred1, pred2))  # Both wrong
                
                # McNemar's chi-square statistic
                if b + c > 0:
                    chi_square = (abs(b - c) - 1) ** 2 / (b + c)
                else:
                    chi_square = 0
                
                significance_results[f"{model1}_vs_{model2}"] = {
                    "chi_square": chi_square,
                    "p_value": self._chi_square_p_value(chi_square, 1),
                    "significant": chi_square > 3.841  # 0.05 significance level
                }
        
        return significance_results
    
    def _get_model_predictions(self, model_name: str, test_data: pd.DataFrame) -> List[bool]:
        """Get binary predictions for a model."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        predictions = []
        for _, row in test_data.iterrows():
            chosen_score = self._get_reward_score(model, tokenizer, row['chosen'])
            rejected_score = self._get_reward_score(model, tokenizer, row['rejected'])
            predictions.append(chosen_score > rejected_score)
        
        return predictions
    
    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Calculate p-value for chi-square statistic (simplified)."""
        # This is a very simplified approximation
        # In practice, use scipy.stats.chi2.sf(chi_square, df)
        if chi_square < 0.455:
            return 0.5
        elif chi_square < 1.323:
            return 0.25
        elif chi_square < 2.706:
            return 0.1
        elif chi_square < 3.841:
            return 0.05
        elif chi_square < 5.024:
            return 0.025
        elif chi_square < 6.635:
            return 0.01
        else:
            return 0.005
    
    def create_comparison_report(self, output_path: str = None) -> str:
        """Create a comprehensive comparison report."""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run compare_models_on_dataset first.")
        
        logger.info("📋 Creating model comparison report...")
        
        results = self.comparison_results
        model_names = results["model_names"]
        
        report = f"""
# Model Comparison Report

## Summary
- **Models Compared**: {', '.join(model_names)}
- **Test Dataset Size**: {results['test_data_size']} samples
- **Comparison Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Comparison
"""
        
        # Performance ranking
        if "performance_ranking" in results["comparison_metrics"]:
            report += "\n### Preference Accuracy Ranking\n"
            for i, (model_name, accuracy) in enumerate(results["comparison_metrics"]["performance_ranking"]):
                report += f"{i+1}. **{model_name}**: {accuracy:.4f}\n"
        
        # Individual model results
        report += "\n## Individual Model Results\n"
        for model_name in model_names:
            model_results = results["individual_results"][model_name]
            report += f"\n### {model_name}\n"
            
            if "preference_accuracy" in model_results:
                report += f"- **Preference Accuracy**: {model_results['preference_accuracy']:.4f}\n"
            
            if "ranking_metrics" in model_results:
                rm = model_results["ranking_metrics"]
                report += f"- **NDCG**: {rm['ndcg']:.4f}\n"
                report += f"- **MRR**: {rm['mrr']:.4f}\n"
                report += f"- **Kendall's Tau**: {rm['kendall_tau']:.4f}\n"
            
            if "score_statistics" in model_results:
                ss = model_results["score_statistics"]
                report += f"- **Chosen Score Mean**: {ss['chosen_mean']:.4f}\n"
                report += f"- **Rejected Score Mean**: {ss['rejected_mean']:.4f}\n"
                report += f"- **Score Difference Mean**: {ss['score_difference_mean']:.4f}\n"
        
        # Agreement matrix
        if "agreement_matrix" in results["comparison_metrics"]:
            report += "\n## Model Agreement Matrix\n"
            agreement_matrix = results["comparison_metrics"]["agreement_matrix"]
            report += agreement_matrix.to_string()
        
        # Statistical significance
        if "significance_tests" in results["comparison_metrics"]:
            report += "\n\n## Statistical Significance Tests\n"
            for test_name, test_result in results["comparison_metrics"]["significance_tests"].items():
                significance = "**Significant**" if test_result["significant"] else "Not significant"
                report += f"- **{test_name}**: χ² = {test_result['chi_square']:.3f}, p = {test_result['p_value']:.3f} ({significance})\n"
        
        # Recommendations
        report += "\n## Recommendations\n"
        
        # Find best performing model
        if "performance_ranking" in results["comparison_metrics"]:
            best_model = results["comparison_metrics"]["performance_ranking"][0][0]
            report += f"- **Best Performing Model**: {best_model}\n"
        
        # Check for significant differences
        if "significance_tests" in results["comparison_metrics"]:
            significant_differences = [
                test_name for test_name, test_result in results["comparison_metrics"]["significance_tests"].items()
                if test_result["significant"]
            ]
            if significant_differences:
                report += f"- **Significant Differences Found**: {len(significant_differences)} model pairs show significant performance differences\n"
            else:
                report += "- **No Significant Differences**: All models perform similarly\n"
        
        report += "\n---\n*Report generated by ModelComparisonTool*"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📊 Comparison report saved to: {output_path}")
        
        return report
    
    def create_comparison_visualizations(self, output_dir: str = None):
        """Create visualizations for model comparison."""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run compare_models_on_dataset first.")
        
        logger.info("📊 Creating comparison visualizations...")
        
        results = self.comparison_results
        model_names = results["model_names"]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        # 1. Performance comparison
        ax1 = axes[0, 0]
        accuracies = [results["individual_results"][name].get("preference_accuracy", 0) for name in model_names]
        bars = ax1.bar(model_names, accuracies, alpha=0.7)
        ax1.set_title('Preference Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Agreement heatmap
        ax2 = axes[0, 1]
        agreement_matrix = results["comparison_metrics"]["agreement_matrix"]
        sns.heatmap(agreement_matrix, annot=True, cmap='Blues', ax=ax2)
        ax2.set_title('Model Agreement Matrix')
        
        # 3. Score distribution comparison
        ax3 = axes[1, 0]
        for model_name in model_names:
            chosen_scores = [results["individual_results"][model_name]["score_statistics"]["chosen_mean"]]
            rejected_scores = [results["individual_results"][model_name]["score_statistics"]["rejected_mean"]]
            ax3.scatter(chosen_scores, rejected_scores, label=model_name, s=100, alpha=0.7)
        
        ax3.set_xlabel('Chosen Score Mean')
        ax3.set_ylabel('Rejected Score Mean')
        ax3.set_title('Score Distribution Comparison')
        ax3.legend()
        
        # Add diagonal line
        min_val = min(ax3.get_xlim()[0], ax3.get_ylim()[0])
        max_val = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # 4. Ranking metrics comparison
        ax4 = axes[1, 1]
        if "ranking_metrics" in results["individual_results"][model_names[0]]:
            metrics = ['ndcg', 'mrr', 'kendall_tau']
            x = np.arange(len(metrics))
            width = 0.8 / len(model_names)
            
            for i, model_name in enumerate(model_names):
                values = [results["individual_results"][model_name]["ranking_metrics"][metric] for metric in metrics]
                ax4.bar(x + i * width, values, width, label=model_name, alpha=0.7)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Score')
            ax4.set_title('Ranking Metrics Comparison')
            ax4.set_xticks(x + width * (len(model_names) - 1) / 2)
            ax4.set_xticklabels(metrics)
            ax4.legend()
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualizations saved to: {output_path / 'model_comparison.png'}")
        
        plt.show()
    
    def _get_reward_score(self, model, tokenizer, text: str) -> float:
        """Get reward score for a given text using the specified model and tokenizer."""
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = model(**inputs)
            score = outputs.logits.item()
            return score
    
    def _calculate_ndcg(self, chosen_scores: List[float], rejected_scores: List[float]) -> float:
        """Calculate NDCG for a list of score pairs."""
        dcg = 0
        idcg = 0
        
        for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
            if chosen_score > rejected_score:
                dcg += 1.0 / np.log2(2)
                idcg += 1.0 / np.log2(2)
            else:
                dcg += 0.0 / np.log2(2)
                idcg += 1.0 / np.log2(2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_mrr(self, chosen_scores: List[float], rejected_scores: List[float]) -> float:
        """Calculate MRR for a list of score pairs."""
        reciprocal_ranks = []
        
        for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
            if chosen_score > rejected_score:
                reciprocal_ranks.append(1.0)
            else:
                reciprocal_ranks.append(0.5)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    def _calculate_kendall_tau(self, chosen_scores: List[float], rejected_scores: List[float]) -> float:
        """Calculate Kendall's Tau for a list of score pairs."""
        score_differences = [c - r for c, r in zip(chosen_scores, rejected_scores)]
        preferences = [1 if c > r else 0 for c, r in zip(chosen_scores, rejected_scores)]
        
        if len(set(preferences)) < 2:
            return 0.0
        
        correlation = np.corrcoef(score_differences, preferences)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _detect_bias(self, test_data: pd.DataFrame, chosen_scores: List[float], 
                    rejected_scores: List[float]) -> Dict[str, float]:
        """Detect bias in model predictions."""
        # Simple bias detection based on length correlation
        chosen_lengths = [len(row['chosen']) for _, row in test_data.iterrows()]
        rejected_lengths = [len(row['rejected']) for _, row in test_data.iterrows()]
        
        length_correlation = np.corrcoef(chosen_scores, chosen_lengths)[0, 1] if len(chosen_scores) > 1 else 0
        
        return {
            "length_bias": length_correlation if not np.isnan(length_correlation) else 0
        }
    
    def _calculate_consistency(self, model, tokenizer, test_data: pd.DataFrame, 
                             num_runs: int = 3) -> float:
        """Calculate model consistency across multiple runs."""
        consistency_scores = []
        
        for run in range(num_runs):
            run_scores = []
            for _, row in test_data.iterrows():
                chosen_score = self._get_reward_score(model, tokenizer, row['chosen'])
                rejected_score = self._get_reward_score(model, tokenizer, row['rejected'])
                run_scores.append(chosen_score - rejected_score)
            consistency_scores.append(run_scores)
        
        # Calculate coefficient of variation
        consistency_scores = np.array(consistency_scores)
        cv_scores = np.std(consistency_scores, axis=0) / (np.mean(consistency_scores, axis=0) + 1e-8)
        avg_cv = np.mean(cv_scores)
        
        return 1.0 - avg_cv  # Higher is better 