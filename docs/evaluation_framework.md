# Reward Model Evaluation Framework

This document describes the comprehensive evaluation framework for reward models, including custom metrics, bias detection, human evaluation, and model comparison tools.

## Overview

The evaluation framework provides a complete suite of tools for assessing reward model performance:

1. **Custom Evaluation Metrics** - Preference accuracy, ranking metrics, and quality analysis
2. **Bias Detection & Analysis** - Identify and measure potential biases in model outputs
3. **Human Evaluation Framework** - Compare model outputs with human judgments
4. **Model Comparison Tools** - Compare multiple models side by side
5. **Evaluation Reports & Visualizations** - Comprehensive reporting and analysis

## Quick Start

### Basic Evaluation

```bash
# Evaluate a single model
python src/evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv

# With custom output directory
python src/evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --output-dir ./my_results

# With human evaluation setup
python src/evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --human-eval

# Model comparison
python src/evaluate_reward_model.py --model-path ./models/reward_model --test-data data/test.csv --compare-models model1:./models/model1,model2:./models/model2
```

### Programmatic Usage

```python
from evaluation_metrics import RewardModelEvaluator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./models/reward_model")
tokenizer = AutoTokenizer.from_pretrained("./models/reward_model")

# Create evaluator
evaluator = RewardModelEvaluator(model, tokenizer, device="cuda")

# Load test data
test_data = pd.read_csv("data/test.csv")

# Run comprehensive evaluation
report = evaluator.generate_evaluation_report(test_data, "evaluation_report.json")

# Create visualizations
evaluator.create_visualizations(test_data, "./results")
```

## Evaluation Metrics

### 1. Preference Accuracy

Measures how often the model correctly identifies the better response:

```python
accuracy_results = evaluator.evaluate_preference_accuracy(test_data)
print(f"Preference Accuracy: {accuracy_results['preference_accuracy']:.4f}")
```

**Metrics:**
- `preference_accuracy`: Percentage of correct preference predictions
- `correct_predictions`: Number of correct predictions
- `total_predictions`: Total number of predictions
- `avg_confidence`: Average confidence in predictions
- `confidence_std`: Standard deviation of confidence

### 2. Ranking Metrics

Evaluates the model's ranking performance:

```python
ranking_results = evaluator.evaluate_ranking_metrics(test_data)
print(f"NDCG: {ranking_results['ndcg']:.4f}")
print(f"MRR: {ranking_results['mrr']:.4f}")
print(f"Kendall's Tau: {ranking_results['kendall_tau']:.4f}")
```

**Metrics:**
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks
- **Kendall's Tau**: Correlation between predicted and actual rankings

### 3. Bias Detection

Identifies potential biases in the model:

```python
bias_results = evaluator.evaluate_bias_detection(test_data)
print(f"Overall Bias Score: {bias_results['overall_bias']:.4f}")
```

**Bias Types Detected:**
- **Gender Bias**: He/she, his/her, man/woman references
- **Race Bias**: Racial and ethnic references
- **Age Bias**: Age-related terms
- **Profession Bias**: Professional role references
- **Sentiment Bias**: Positive/negative sentiment terms

### 4. Quality Distribution Analysis

Analyzes the distribution of reward scores:

```python
quality_results = evaluator.evaluate_response_quality_distribution(test_data)
print(f"Chosen Score Mean: {quality_results['chosen_scores']['mean']:.4f}")
print(f"Rejected Score Mean: {quality_results['rejected_scores']['mean']:.4f}")
```

**Statistics:**
- Score distributions for chosen and rejected responses
- Score difference analysis
- Length correlation analysis

### 5. Model Consistency

Evaluates model consistency across multiple runs:

```python
consistency_results = evaluator.evaluate_model_consistency(test_data)
print(f"Consistency Score: {consistency_results['consistency_score']:.4f}")
```

**Metrics:**
- **Consistency Score**: Higher values indicate more consistent predictions
- **Average Run Correlation**: Correlation between different evaluation runs
- **Coefficient of Variation**: Measure of prediction variability

## Human Evaluation Framework

### Setup

```python
from human_evaluation import HumanEvaluationFramework

# Create framework
config = {
    "evaluation_questions": [
        "Which response is more helpful?",
        "Which response is more accurate?",
        "Which response is more appropriate?"
    ],
    "evaluation_scale": 5,
    "min_evaluations_per_sample": 3
}

human_evaluator = HumanEvaluationFramework(config)
```

### Create Evaluation Dataset

```python
# Generate model outputs for evaluation
model_outputs = [
    {
        "prompt": "What is machine learning?",
        "response_a": "Machine learning is a subset of AI...",
        "response_b": "ML is when computers learn...",
        "model_preference": "A",
        "model_confidence": 0.8
    }
]

# Create evaluation dataset
evaluation_dataset = human_evaluator.create_evaluation_dataset(model_outputs, num_samples=50)
```

### Launch Evaluation Interface

```python
# Create Gradio interface
interface = human_evaluator.create_gradio_interface(evaluation_dataset)

# Launch the interface
interface.launch(share=False)
```

### Analyze Results

```python
# Load evaluation data
human_evaluator.load_evaluation_data("evaluation_results.json")

# Analyze results
results = human_evaluator.analyze_human_evaluations(
    human_evaluator.evaluation_data, 
    model_predictions_df
)

# Generate report
report = human_evaluator.create_evaluation_report(results, "human_eval_report.md")
```

## Model Comparison Tools

### Compare Multiple Models

```python
from model_comparison import ModelComparisonTool

# Prepare models and tokenizers
models = {
    "model_a": model_a,
    "model_b": model_b,
    "model_c": model_c
}

tokenizers = {
    "model_a": tokenizer_a,
    "model_b": tokenizer_b,
    "model_c": tokenizer_c
}

# Create comparison tool
comparison_tool = ModelComparisonTool(models, tokenizers, device="cuda")

# Run comparison
results = comparison_tool.compare_models_on_dataset(test_data)

# Generate report
report = comparison_tool.create_comparison_report("comparison_report.md")

# Create visualizations
comparison_tool.create_comparison_visualizations("./comparison_results")
```

### Comparison Metrics

**Agreement Matrix:**
- Shows agreement rates between all model pairs
- Values range from 0 (no agreement) to 1 (perfect agreement)

**Performance Ranking:**
- Ranks models by preference accuracy
- Includes confidence intervals

**Statistical Significance:**
- McNemar's test for paired comparisons
- Identifies significant performance differences

## Configuration

### Evaluation Configuration

```json
{
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
```

### Custom Bias Indicators

```python
custom_bias_indicators = {
    "technical_terms": ["algorithm", "neural network", "deep learning"],
    "domain_specific": ["medical", "legal", "financial"],
    "formality": ["formal", "informal", "casual", "professional"]
}

bias_results = evaluator.evaluate_bias_detection(test_data, custom_bias_indicators)
```

## Output Files

### Evaluation Reports

1. **evaluation_report.json**: Comprehensive evaluation results
2. **evaluation_plots.png**: Visualization plots
3. **human_evaluation_dataset.csv**: Dataset for human evaluation
4. **model_comparison_report.md**: Model comparison report
5. **comparison_plots.png**: Model comparison visualizations

### Report Structure

```json
{
  "preference_accuracy": {
    "preference_accuracy": 0.85,
    "correct_predictions": 850,
    "total_predictions": 1000,
    "avg_confidence": 0.72,
    "confidence_std": 0.15
  },
  "ranking_metrics": {
    "ndcg": 0.92,
    "mrr": 0.88,
    "kendall_tau": 0.76
  },
  "bias_detection": {
    "gender": {"bias_score": 0.05, "sample_count": 150},
    "race": {"bias_score": 0.03, "sample_count": 80},
    "overall_bias": 0.04
  },
  "overall_score": 0.82
}
```

## Best Practices

### 1. Test Data Preparation

- Ensure balanced representation of different response types
- Include edge cases and challenging examples
- Validate data quality before evaluation

### 2. Evaluation Strategy

- Use multiple evaluation metrics for comprehensive assessment
- Include human evaluation for subjective aspects
- Compare against baseline models

### 3. Bias Analysis

- Regularly check for new bias indicators
- Monitor bias scores over time
- Consider domain-specific bias indicators

### 4. Human Evaluation

- Provide clear evaluation guidelines
- Use multiple evaluators for reliability
- Analyze evaluator consistency

### 5. Model Comparison

- Use the same test data for fair comparison
- Consider statistical significance
- Document model differences

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use gradient checkpointing
2. **Slow Evaluation**: Use GPU acceleration and batch processing
3. **Import Errors**: Ensure all dependencies are installed
4. **Data Format Issues**: Validate CSV/JSON format and required columns

### Performance Tips

1. **GPU Acceleration**: Use CUDA for faster evaluation
2. **Batch Processing**: Process data in batches for large datasets
3. **Parallel Evaluation**: Use multiple processes for human evaluation
4. **Caching**: Cache model predictions for repeated evaluations

## Examples

### Complete Evaluation Pipeline

```python
# 1. Load model and data
model = AutoModelForSequenceClassification.from_pretrained("./models/reward_model")
tokenizer = AutoTokenizer.from_pretrained("./models/reward_model")
test_data = pd.read_csv("data/test.csv")

# 2. Create evaluator
evaluator = RewardModelEvaluator(model, tokenizer, device="cuda")

# 3. Run comprehensive evaluation
report = evaluator.generate_evaluation_report(test_data, "evaluation_report.json")

# 4. Create visualizations
evaluator.create_visualizations(test_data, "./results")

# 5. Set up human evaluation
human_evaluator = HumanEvaluationFramework()
interface = human_evaluator.create_gradio_interface(evaluation_dataset)
interface.launch()

# 6. Compare with other models
comparison_tool = ModelComparisonTool(models, tokenizers, device="cuda")
comparison_results = comparison_tool.compare_models_on_dataset(test_data)
```

### Custom Evaluation Script

```python
def custom_evaluation_pipeline():
    # Load configuration
    with open("evaluation_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize evaluator app
    evaluator_app = RewardModelEvaluatorApp(config)
    
    # Load model and data
    evaluator_app.load_model("./models/reward_model")
    test_data = evaluator_app.load_test_data("data/test.csv")
    
    # Run evaluation
    results = evaluator_app.run_comprehensive_evaluation(test_data)
    
    # Print summary
    evaluator_app.print_summary()
    
    return results

if __name__ == "__main__":
    results = custom_evaluation_pipeline()
```

## Integration with Training

The evaluation framework can be integrated into the training pipeline:

```python
# In training script
from evaluation_metrics import RewardModelEvaluator

# After training
evaluator = RewardModelEvaluator(model, tokenizer, device)
report = evaluator.generate_evaluation_report(test_data)

# Save evaluation results
with open("training_evaluation.json", "w") as f:
    json.dump(report, f, indent=2)
```

This comprehensive evaluation framework provides everything needed to thoroughly assess reward model performance and identify areas for improvement. 