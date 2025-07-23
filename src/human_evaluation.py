import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import random
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class HumanEvaluationFramework:
    """Framework for human evaluation of reward model outputs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.evaluation_data = []
        self.results = {}
        
        # Default configuration
        self.evaluation_questions = self.config.get("evaluation_questions", [
            "Which response is more helpful?",
            "Which response is more accurate?",
            "Which response is more appropriate?",
            "Which response would you prefer to receive?"
        ])
        
        self.evaluation_scale = self.config.get("evaluation_scale", 5)  # 1-5 scale
        self.min_evaluations_per_sample = self.config.get("min_evaluations_per_sample", 3)
    
    def create_evaluation_dataset(self, model_outputs: List[Dict[str, Any]], 
                                 num_samples: int = 50) -> pd.DataFrame:
        """Create a dataset for human evaluation."""
        logger.info("📝 Creating human evaluation dataset...")
        
        # Sample from model outputs
        if len(model_outputs) > num_samples:
            selected_outputs = random.sample(model_outputs, num_samples)
        else:
            selected_outputs = model_outputs
        
        evaluation_samples = []
        
        for i, output in enumerate(selected_outputs):
            sample = {
                "sample_id": f"sample_{i:04d}",
                "prompt": output.get("prompt", ""),
                "response_a": output.get("response_a", ""),
                "response_b": output.get("response_b", ""),
                "model_preference": output.get("model_preference", "A"),
                "model_confidence": output.get("model_confidence", 0.5),
                "created_at": datetime.now().isoformat()
            }
            evaluation_samples.append(sample)
        
        df = pd.DataFrame(evaluation_samples)
        logger.info(f"✅ Created evaluation dataset with {len(df)} samples")
        
        return df
    
    def create_gradio_interface(self, evaluation_data: pd.DataFrame) -> gr.Interface:
        """Create a Gradio interface for human evaluation."""
        logger.info("🖥️ Creating Gradio evaluation interface...")
        
        def evaluate_sample(sample_id: str, question: str, rating_a: int, rating_b: int, 
                          comments: str, evaluator_id: str) -> Tuple[str, str, str, str, str, str]:
            """Process a single evaluation."""
            evaluation = {
                "sample_id": sample_id,
                "question": question,
                "rating_a": rating_a,
                "rating_b": rating_b,
                "human_preference": "A" if rating_a > rating_b else "B" if rating_b > rating_a else "Tie",
                "preference_strength": abs(rating_a - rating_b),
                "comments": comments,
                "evaluator_id": evaluator_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.evaluation_data.append(evaluation)
            
            return (
                f"Evaluation saved for sample {sample_id}",
                "Sample ID",
                "Which response is better?",
                3, 3,
                ""
            )
        
        def get_next_sample() -> Tuple[str, str, str, str, str, str, str, str]:
            """Get the next sample for evaluation."""
            # This would typically load from a database or file
            # For now, return a sample from the evaluation data
            if len(evaluation_data) > 0:
                sample = evaluation_data.iloc[0]
                return (
                    sample["sample_id"],
                    sample["prompt"],
                    sample["response_a"],
                    sample["response_b"],
                    sample["model_preference"],
                    f"{sample['model_confidence']:.3f}",
                    "Which response is more helpful?",
                    "evaluator_001"
                )
            else:
                return ("", "", "", "", "", "", "Which response is more helpful?", "evaluator_001")
        
        # Create the interface
        with gr.Blocks(title="Reward Model Human Evaluation") as interface:
            gr.Markdown("# 🤖 Reward Model Human Evaluation")
            gr.Markdown("Please evaluate the quality of AI responses compared to human preferences.")
            
            with gr.Row():
                with gr.Column():
                    sample_id = gr.Textbox(label="Sample ID", interactive=False)
                    prompt = gr.Textbox(label="Prompt", lines=3, interactive=False)
                    
                with gr.Column():
                    model_pref = gr.Textbox(label="Model Preference", interactive=False)
                    model_conf = gr.Textbox(label="Model Confidence", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Response A")
                    response_a = gr.Textbox(label="Response A", lines=5, interactive=False)
                    rating_a = gr.Slider(minimum=1, maximum=self.evaluation_scale, value=3, 
                                       step=1, label="Rating for Response A")
                
                with gr.Column():
                    gr.Markdown("### Response B")
                    response_b = gr.Textbox(label="Response B", lines=5, interactive=False)
                    rating_b = gr.Slider(minimum=1, maximum=self.evaluation_scale, value=3, 
                                       step=1, label="Rating for Response B")
            
            with gr.Row():
                question = gr.Dropdown(choices=self.evaluation_questions, 
                                     value=self.evaluation_questions[0],
                                     label="Evaluation Question")
                evaluator_id = gr.Textbox(label="Evaluator ID", value="evaluator_001")
            
            comments = gr.Textbox(label="Comments (optional)", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("Submit Evaluation", variant="primary")
                next_btn = gr.Button("Next Sample")
            
            status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            submit_btn.click(
                fn=evaluate_sample,
                inputs=[sample_id, question, rating_a, rating_b, comments, evaluator_id],
                outputs=[status, sample_id, question, rating_a, rating_b, comments]
            )
            
            next_btn.click(
                fn=get_next_sample,
                outputs=[sample_id, prompt, response_a, response_b, model_pref, model_conf, question, evaluator_id]
            )
        
        return interface
    
    def analyze_human_evaluations(self, evaluation_data: List[Dict[str, Any]], 
                                 model_predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze human evaluation results and compare with model predictions."""
        logger.info("📊 Analyzing human evaluation results...")
        
        if not evaluation_data:
            logger.warning("No evaluation data provided")
            return {}
        
        # Convert to DataFrame
        eval_df = pd.DataFrame(evaluation_data)
        
        # Merge with model predictions
        merged_data = eval_df.merge(model_predictions, on="sample_id", how="inner")
        
        # Calculate agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(merged_data)
        
        # Calculate correlation metrics
        correlation_metrics = self._calculate_correlation_metrics(merged_data)
        
        # Calculate bias analysis
        bias_analysis = self._analyze_human_bias(merged_data)
        
        # Calculate evaluator consistency
        evaluator_consistency = self._analyze_evaluator_consistency(eval_df)
        
        results = {
            "agreement_metrics": agreement_metrics,
            "correlation_metrics": correlation_metrics,
            "bias_analysis": bias_analysis,
            "evaluator_consistency": evaluator_consistency,
            "total_evaluations": len(eval_df),
            "unique_samples": eval_df["sample_id"].nunique(),
            "unique_evaluators": eval_df["evaluator_id"].nunique()
        }
        
        logger.info(f"✅ Analysis completed: {results['total_evaluations']} evaluations")
        
        return results
    
    def _calculate_agreement_metrics(self, merged_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate agreement between human and model preferences."""
        # Simple agreement (exact match)
        exact_agreement = (merged_data["human_preference"] == merged_data["model_preference"]).mean()
        
        # Agreement considering ties
        human_prefs = merged_data["human_preference"].values
        model_prefs = merged_data["model_preference"].values
        
        # Count agreements excluding ties
        non_tie_mask = (human_prefs != "Tie") & (model_prefs != "Tie")
        if non_tie_mask.sum() > 0:
            agreement_no_ties = (human_prefs[non_tie_mask] == model_prefs[non_tie_mask]).mean()
        else:
            agreement_no_ties = 0.0
        
        # Agreement strength correlation
        strength_correlation = np.corrcoef(
            merged_data["preference_strength"], 
            merged_data["model_confidence"]
        )[0, 1] if len(merged_data) > 1 else 0.0
        
        return {
            "exact_agreement": exact_agreement,
            "agreement_no_ties": agreement_no_ties,
            "strength_correlation": strength_correlation if not np.isnan(strength_correlation) else 0.0
        }
    
    def _calculate_correlation_metrics(self, merged_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation between human ratings and model scores."""
        correlations = {}
        
        # Correlation between human preference strength and model confidence
        if len(merged_data) > 1:
            strength_conf_corr = np.corrcoef(
                merged_data["preference_strength"], 
                merged_data["model_confidence"]
            )[0, 1]
            correlations["preference_strength_confidence_corr"] = strength_conf_corr if not np.isnan(strength_conf_corr) else 0.0
        
        # Average rating correlation
        avg_ratings = merged_data.groupby("sample_id")[["rating_a", "rating_b"]].mean()
        if len(avg_ratings) > 1:
            rating_corr = np.corrcoef(avg_ratings["rating_a"], avg_ratings["rating_b"])[0, 1]
            correlations["avg_rating_correlation"] = rating_corr if not np.isnan(rating_corr) else 0.0
        
        return correlations
    
    def _analyze_human_bias(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential biases in human evaluations."""
        bias_analysis = {}
        
        # Response order bias (A vs B preference)
        a_preferences = (merged_data["human_preference"] == "A").sum()
        b_preferences = (merged_data["human_preference"] == "B").sum()
        tie_preferences = (merged_data["human_preference"] == "Tie").sum()
        
        total_preferences = a_preferences + b_preferences + tie_preferences
        
        bias_analysis["response_order_bias"] = {
            "a_preference_rate": a_preferences / total_preferences if total_preferences > 0 else 0,
            "b_preference_rate": b_preferences / total_preferences if total_preferences > 0 else 0,
            "tie_rate": tie_preferences / total_preferences if total_preferences > 0 else 0
        }
        
        # Rating bias (systematic rating differences)
        avg_rating_a = merged_data["rating_a"].mean()
        avg_rating_b = merged_data["rating_b"].mean()
        bias_analysis["rating_bias"] = {
            "avg_rating_a": avg_rating_a,
            "avg_rating_b": avg_rating_b,
            "rating_difference": avg_rating_a - avg_rating_b
        }
        
        return bias_analysis
    
    def _analyze_evaluator_consistency(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency between different evaluators."""
        if eval_df["evaluator_id"].nunique() < 2:
            return {"consistency_score": 1.0, "evaluator_count": 1}
        
        # Calculate agreement between evaluators for the same samples
        evaluator_agreements = []
        
        for sample_id in eval_df["sample_id"].unique():
            sample_evals = eval_df[eval_df["sample_id"] == sample_id]
            if len(sample_evals) >= 2:
                # Calculate pairwise agreement
                preferences = sample_evals["human_preference"].values
                agreements = []
                for i in range(len(preferences)):
                    for j in range(i + 1, len(preferences)):
                        agreements.append(preferences[i] == preferences[j])
                
                if agreements:
                    evaluator_agreements.append(np.mean(agreements))
        
        consistency_score = np.mean(evaluator_agreements) if evaluator_agreements else 1.0
        
        return {
            "consistency_score": consistency_score,
            "evaluator_count": eval_df["evaluator_id"].nunique(),
            "avg_agreements_per_sample": np.mean(evaluator_agreements) if evaluator_agreements else 1.0
        }
    
    def create_evaluation_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Create a comprehensive human evaluation report."""
        logger.info("📋 Creating human evaluation report...")
        
        report = f"""
# Human Evaluation Report

## Summary
- **Total Evaluations**: {results.get('total_evaluations', 0)}
- **Unique Samples**: {results.get('unique_samples', 0)}
- **Unique Evaluators**: {results.get('unique_evaluators', 0)}

## Agreement Metrics
- **Exact Agreement**: {results.get('agreement_metrics', {}).get('exact_agreement', 0):.3f}
- **Agreement (No Ties)**: {results.get('agreement_metrics', {}).get('agreement_no_ties', 0):.3f}
- **Strength Correlation**: {results.get('agreement_metrics', {}).get('strength_correlation', 0):.3f}

## Evaluator Consistency
- **Consistency Score**: {results.get('evaluator_consistency', {}).get('consistency_score', 0):.3f}
- **Evaluator Count**: {results.get('evaluator_consistency', {}).get('evaluator_count', 0)}

## Bias Analysis
- **A Preference Rate**: {results.get('bias_analysis', {}).get('response_order_bias', {}).get('a_preference_rate', 0):.3f}
- **B Preference Rate**: {results.get('bias_analysis', {}).get('response_order_bias', {}).get('b_preference_rate', 0):.3f}
- **Tie Rate**: {results.get('bias_analysis', {}).get('response_order_bias', {}).get('tie_rate', 0):.3f}
- **Rating Difference (A-B)**: {results.get('bias_analysis', {}).get('rating_bias', {}).get('rating_difference', 0):.3f}

## Recommendations
"""
        
        # Add recommendations based on results
        agreement = results.get('agreement_metrics', {}).get('exact_agreement', 0)
        consistency = results.get('evaluator_consistency', {}).get('consistency_score', 0)
        
        if agreement < 0.7:
            report += "- **Low Agreement**: Consider improving model training or evaluation criteria\n"
        
        if consistency < 0.8:
            report += "- **Low Consistency**: Consider providing better evaluation guidelines\n"
        
        if results.get('bias_analysis', {}).get('response_order_bias', {}).get('a_preference_rate', 0) > 0.6:
            report += "- **Order Bias**: Consider randomizing response order\n"
        
        report += "\n---\n*Report generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📊 Report saved to: {output_path}")
        
        return report
    
    def save_evaluation_data(self, output_path: str):
        """Save evaluation data to file."""
        if self.evaluation_data:
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_data, f, indent=2)
            logger.info(f"💾 Evaluation data saved to: {output_path}")
    
    def load_evaluation_data(self, input_path: str):
        """Load evaluation data from file."""
        with open(input_path, 'r') as f:
            self.evaluation_data = json.load(f)
        logger.info(f"📂 Loaded {len(self.evaluation_data)} evaluations from {input_path}")

def create_sample_evaluation_data():
    """Create sample evaluation data for testing."""
    return [
        {
            "sample_id": "sample_0001",
            "question": "Which response is more helpful?",
            "rating_a": 4,
            "rating_b": 2,
            "human_preference": "A",
            "preference_strength": 2,
            "comments": "Response A provides more detailed and actionable advice",
            "evaluator_id": "evaluator_001",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "sample_id": "sample_0001",
            "question": "Which response is more helpful?",
            "rating_a": 3,
            "rating_b": 3,
            "human_preference": "Tie",
            "preference_strength": 0,
            "comments": "Both responses are equally helpful",
            "evaluator_id": "evaluator_002",
            "timestamp": "2024-01-01T10:30:00"
        }
    ] 