import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RewardModelEvaluator:
    """Comprehensive evaluation framework for reward models."""
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.evaluation_results = {}
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_preference_accuracy(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate preference accuracy - how often the model correctly identifies the better response."""
        logger.info("🎯 Evaluating preference accuracy...")
        
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        
        for _, row in test_data.iterrows():
            chosen_text = row['chosen']
            rejected_text = row['rejected']
            
            # Get reward scores
            chosen_score = self._get_reward_score(chosen_text)
            rejected_score = self._get_reward_score(rejected_text)
            
            # Check if model correctly prefers chosen over rejected
            if chosen_score > rejected_score:
                correct_predictions += 1
            
            total_predictions += 1
            
            # Calculate confidence (difference between scores)
            confidence = abs(chosen_score - rejected_score)
            confidence_scores.append(confidence)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            "preference_accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "avg_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores)
        }
        
        logger.info(f"   - Preference Accuracy: {accuracy:.4f}")
        logger.info(f"   - Average Confidence: {results['avg_confidence']:.4f}")
        
        return results
    
    def evaluate_ranking_metrics(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ranking metrics like NDCG, MRR, and Kendall's Tau."""
        logger.info("📊 Evaluating ranking metrics...")
        
        # Calculate reward scores for all responses
        scores = []
        for _, row in test_data.iterrows():
            chosen_score = self._get_reward_score(row['chosen'])
            rejected_score = self._get_reward_score(row['rejected'])
            scores.append((chosen_score, rejected_score))
        
        # Calculate ranking metrics
        ndcg = self._calculate_ndcg(scores)
        mrr = self._calculate_mrr(scores)
        kendall_tau = self._calculate_kendall_tau(scores)
        
        results = {
            "ndcg": ndcg,
            "mrr": mrr,
            "kendall_tau": kendall_tau,
            "avg_score_difference": np.mean([abs(c - r) for c, r in scores])
        }
        
        logger.info(f"   - NDCG: {ndcg:.4f}")
        logger.info(f"   - MRR: {mrr:.4f}")
        logger.info(f"   - Kendall's Tau: {kendall_tau:.4f}")
        
        return results
    
    def evaluate_bias_detection(self, test_data: pd.DataFrame, bias_indicators: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Detect and analyze potential biases in the reward model."""
        logger.info("🔍 Evaluating bias detection...")
        
        if bias_indicators is None:
            bias_indicators = self._get_default_bias_indicators()
        
        bias_results = {}
        
        for bias_type, indicators in bias_indicators.items():
            logger.info(f"   - Analyzing {bias_type} bias...")
            bias_scores = self._analyze_bias(test_data, indicators, bias_type)
            bias_results[bias_type] = bias_scores
        
        # Overall bias summary
        overall_bias = self._calculate_overall_bias(bias_results)
        bias_results["overall_bias"] = overall_bias
        
        logger.info(f"   - Overall Bias Score: {overall_bias:.4f}")
        
        return bias_results
    
    def evaluate_response_quality_distribution(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of reward scores across different response characteristics."""
        logger.info("📈 Analyzing response quality distribution...")
        
        # Calculate scores for all responses
        chosen_scores = []
        rejected_scores = []
        length_differences = []
        
        for _, row in test_data.iterrows():
            chosen_score = self._get_reward_score(row['chosen'])
            rejected_score = self._get_reward_score(row['rejected'])
            
            chosen_scores.append(chosen_score)
            rejected_scores.append(rejected_score)
            
            # Length difference
            length_diff = len(row['chosen']) - len(row['rejected'])
            length_differences.append(length_diff)
        
        # Analyze distributions
        distribution_stats = {
            "chosen_scores": {
                "mean": np.mean(chosen_scores),
                "std": np.std(chosen_scores),
                "min": np.min(chosen_scores),
                "max": np.max(chosen_scores),
                "percentiles": np.percentile(chosen_scores, [25, 50, 75])
            },
            "rejected_scores": {
                "mean": np.mean(rejected_scores),
                "std": np.std(rejected_scores),
                "min": np.min(rejected_scores),
                "max": np.max(rejected_scores),
                "percentiles": np.percentile(rejected_scores, [25, 50, 75])
            },
            "score_differences": {
                "mean": np.mean([c - r for c, r in zip(chosen_scores, rejected_scores)]),
                "std": np.std([c - r for c, r in zip(chosen_scores, rejected_scores)])
            },
            "length_correlation": np.corrcoef(chosen_scores, [abs(ld) for ld in length_differences])[0, 1]
        }
        
        logger.info(f"   - Chosen Score Mean: {distribution_stats['chosen_scores']['mean']:.4f}")
        logger.info(f"   - Rejected Score Mean: {distribution_stats['rejected_scores']['mean']:.4f}")
        logger.info(f"   - Score Difference Mean: {distribution_stats['score_differences']['mean']:.4f}")
        
        return distribution_stats
    
    def evaluate_model_consistency(self, test_data: pd.DataFrame, num_runs: int = 5) -> Dict[str, float]:
        """Evaluate model consistency across multiple runs."""
        logger.info("🔄 Evaluating model consistency...")
        
        consistency_scores = []
        
        for run in range(num_runs):
            run_scores = []
            for _, row in test_data.iterrows():
                chosen_score = self._get_reward_score(row['chosen'])
                rejected_score = self._get_reward_score(row['rejected'])
                run_scores.append(chosen_score - rejected_score)
            consistency_scores.append(run_scores)
        
        # Calculate consistency metrics
        consistency_scores = np.array(consistency_scores)
        
        # Coefficient of variation across runs
        cv_scores = np.std(consistency_scores, axis=0) / (np.mean(consistency_scores, axis=0) + 1e-8)
        avg_cv = np.mean(cv_scores)
        
        # Correlation between runs
        run_correlations = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                corr = np.corrcoef(consistency_scores[i], consistency_scores[j])[0, 1]
                run_correlations.append(corr)
        
        avg_correlation = np.mean(run_correlations)
        
        results = {
            "avg_coefficient_of_variation": avg_cv,
            "avg_run_correlation": avg_correlation,
            "consistency_score": 1.0 - avg_cv  # Higher is better
        }
        
        logger.info(f"   - Consistency Score: {results['consistency_score']:.4f}")
        logger.info(f"   - Average Run Correlation: {avg_correlation:.4f}")
        
        return results
    
    def compare_models(self, other_model, test_data: pd.DataFrame, model_names: List[str] = None) -> Dict[str, Any]:
        """Compare two reward models on the same test data."""
        logger.info("⚖️ Comparing models...")
        
        if model_names is None:
            model_names = ["Model A", "Model B"]
        
        # Evaluate current model
        current_evaluator = RewardModelEvaluator(self.model, self.tokenizer, self.device)
        current_results = current_evaluator.evaluate_preference_accuracy(test_data)
        
        # Evaluate other model
        other_evaluator = RewardModelEvaluator(other_model, self.tokenizer, self.device)
        other_results = other_evaluator.evaluate_preference_accuracy(test_data)
        
        # Compare predictions
        agreement_count = 0
        total_predictions = 0
        
        for _, row in test_data.iterrows():
            # Current model prediction
            current_chosen_score = current_evaluator._get_reward_score(row['chosen'])
            current_rejected_score = current_evaluator._get_reward_score(row['rejected'])
            current_prefers_chosen = current_chosen_score > current_rejected_score
            
            # Other model prediction
            other_chosen_score = other_evaluator._get_reward_score(row['chosen'])
            other_rejected_score = other_evaluator._get_reward_score(row['rejected'])
            other_prefers_chosen = other_chosen_score > other_rejected_score
            
            if current_prefers_chosen == other_prefers_chosen:
                agreement_count += 1
            total_predictions += 1
        
        agreement_rate = agreement_count / total_predictions if total_predictions > 0 else 0
        
        comparison_results = {
            "model_names": model_names,
            "agreement_rate": agreement_rate,
            "current_model": current_results,
            "other_model": other_results,
            "accuracy_difference": current_results["preference_accuracy"] - other_results["preference_accuracy"]
        }
        
        logger.info(f"   - Model Agreement Rate: {agreement_rate:.4f}")
        logger.info(f"   - Accuracy Difference: {comparison_results['accuracy_difference']:.4f}")
        
        return comparison_results
    
    def generate_evaluation_report(self, test_data: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        logger.info("📋 Generating comprehensive evaluation report...")
        
        # Run all evaluations
        report = {
            "preference_accuracy": self.evaluate_preference_accuracy(test_data),
            "ranking_metrics": self.evaluate_ranking_metrics(test_data),
            "bias_detection": self.evaluate_bias_detection(test_data),
            "quality_distribution": self.evaluate_response_quality_distribution(test_data),
            "model_consistency": self.evaluate_model_consistency(test_data)
        }
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(report)
        report["overall_score"] = overall_score
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Evaluation report saved to: {output_path}")
        
        return report
    
    def create_visualizations(self, test_data: pd.DataFrame, output_dir: str = None):
        """Create evaluation visualizations."""
        logger.info("📊 Creating evaluation visualizations...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Get scores for visualization
        chosen_scores = []
        rejected_scores = []
        
        for _, row in test_data.iterrows():
            chosen_scores.append(self._get_reward_score(row['chosen']))
            rejected_scores.append(self._get_reward_score(row['rejected']))
        
        # 1. Score Distribution Plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(chosen_scores, alpha=0.7, label='Chosen', bins=30)
        plt.hist(rejected_scores, alpha=0.7, label='Rejected', bins=30)
        plt.xlabel('Reward Score')
        plt.ylabel('Frequency')
        plt.title('Reward Score Distribution')
        plt.legend()
        
        # 2. Score Difference Plot
        plt.subplot(2, 2, 2)
        score_differences = [c - r for c, r in zip(chosen_scores, rejected_scores)]
        plt.hist(score_differences, bins=30, alpha=0.7)
        plt.xlabel('Score Difference (Chosen - Rejected)')
        plt.ylabel('Frequency')
        plt.title('Score Difference Distribution')
        
        # 3. Scatter Plot
        plt.subplot(2, 2, 3)
        plt.scatter(rejected_scores, chosen_scores, alpha=0.6)
        plt.plot([min(rejected_scores), max(rejected_scores)], 
                [min(rejected_scores), max(rejected_scores)], 'r--', label='Equal')
        plt.xlabel('Rejected Score')
        plt.ylabel('Chosen Score')
        plt.title('Chosen vs Rejected Scores')
        plt.legend()
        
        # 4. Length vs Score Plot
        plt.subplot(2, 2, 4)
        chosen_lengths = [len(row['chosen']) for _, row in test_data.iterrows()]
        rejected_lengths = [len(row['rejected']) for _, row in test_data.iterrows()]
        
        plt.scatter(chosen_lengths, chosen_scores, alpha=0.6, label='Chosen')
        plt.scatter(rejected_lengths, rejected_scores, alpha=0.6, label='Rejected')
        plt.xlabel('Text Length')
        plt.ylabel('Reward Score')
        plt.title('Score vs Text Length')
        plt.legend()
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualizations saved to: {output_path / 'evaluation_plots.png'}")
        
        plt.show()
    
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
    
    def _calculate_ndcg(self, scores: List[Tuple[float, float]]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # For binary preference, NDCG is simplified
        # We consider the chosen response as the "relevant" item
        dcg = 0
        idcg = 0
        
        for chosen_score, rejected_score in scores:
            # DCG: sum of (2^relevance - 1) / log2(rank + 1)
            if chosen_score > rejected_score:
                dcg += 1.0 / np.log2(2)  # rank 1
                idcg += 1.0 / np.log2(2)  # ideal case
            else:
                dcg += 0.0 / np.log2(2)  # rank 1
                idcg += 1.0 / np.log2(2)  # ideal case
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_mrr(self, scores: List[Tuple[float, float]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for chosen_score, rejected_score in scores:
            if chosen_score > rejected_score:
                reciprocal_ranks.append(1.0)  # rank 1
            else:
                reciprocal_ranks.append(0.5)  # rank 2
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    def _calculate_kendall_tau(self, scores: List[Tuple[float, float]]) -> float:
        """Calculate Kendall's Tau correlation."""
        # For binary preferences, we calculate correlation between
        # the preference direction and the score difference
        score_differences = [c - r for c, r in scores]
        preferences = [1 if c > r else 0 for c, r in scores]
        
        if len(set(preferences)) < 2:
            return 0.0
        
        # Calculate correlation
        correlation = np.corrcoef(score_differences, preferences)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _get_default_bias_indicators(self) -> Dict[str, List[str]]:
        """Get default bias indicators for different types of bias."""
        return {
            "gender": ["he", "she", "his", "her", "him", "man", "woman", "male", "female"],
            "race": ["black", "white", "asian", "hispanic", "african", "caucasian"],
            "age": ["young", "old", "elderly", "teenager", "senior"],
            "profession": ["doctor", "nurse", "teacher", "engineer", "lawyer", "artist"],
            "sentiment": ["good", "bad", "excellent", "terrible", "amazing", "awful"]
        }
    
    def _analyze_bias(self, test_data: pd.DataFrame, indicators: List[str], bias_type: str) -> Dict[str, float]:
        """Analyze bias for a specific type."""
        bias_scores = []
        
        for _, row in test_data.iterrows():
            chosen_text = row['chosen'].lower()
            rejected_text = row['rejected'].lower()
            
            # Count indicator occurrences
            chosen_indicators = sum(1 for indicator in indicators if indicator in chosen_text)
            rejected_indicators = sum(1 for indicator in indicators if indicator in rejected_text)
            
            if chosen_indicators > 0 or rejected_indicators > 0:
                chosen_score = self._get_reward_score(row['chosen'])
                rejected_score = self._get_reward_score(row['rejected'])
                
                # Calculate bias score
                if chosen_indicators > rejected_indicators:
                    bias_score = chosen_score - rejected_score
                elif rejected_indicators > chosen_indicators:
                    bias_score = rejected_score - chosen_score
                else:
                    bias_score = 0
                
                bias_scores.append(bias_score)
        
        if not bias_scores:
            return {"bias_score": 0.0, "sample_count": 0}
        
        return {
            "bias_score": np.mean(bias_scores),
            "bias_std": np.std(bias_scores),
            "sample_count": len(bias_scores)
        }
    
    def _calculate_overall_bias(self, bias_results: Dict[str, Any]) -> float:
        """Calculate overall bias score."""
        bias_scores = []
        
        for bias_type, results in bias_results.items():
            if bias_type != "overall_bias" and isinstance(results, dict):
                bias_scores.append(abs(results.get("bias_score", 0)))
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    def _calculate_overall_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall evaluation score."""
        scores = []
        
        # Preference accuracy (weight: 0.4)
        if "preference_accuracy" in report:
            scores.append(report["preference_accuracy"]["preference_accuracy"] * 0.4)
        
        # Ranking metrics (weight: 0.3)
        if "ranking_metrics" in report:
            ranking_score = (
                report["ranking_metrics"]["ndcg"] * 0.5 +
                report["ranking_metrics"]["mrr"] * 0.3 +
                report["ranking_metrics"]["kendall_tau"] * 0.2
            ) * 0.3
            scores.append(ranking_score)
        
        # Model consistency (weight: 0.2)
        if "model_consistency" in report:
            scores.append(report["model_consistency"]["consistency_score"] * 0.2)
        
        # Bias (weight: 0.1, inverted - lower bias is better)
        if "bias_detection" in report and "overall_bias" in report["bias_detection"]:
            bias_penalty = min(report["bias_detection"]["overall_bias"], 1.0) * 0.1
            scores.append((1.0 - bias_penalty) * 0.1)
        
        return sum(scores) if scores else 0.0 