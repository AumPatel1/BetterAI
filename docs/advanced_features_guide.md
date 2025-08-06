# Advanced Features Guide

This guide covers the advanced features added to BetterAI, including hyperparameter optimization, advanced training capabilities, and production deployment.

## Table of Contents

1. [Hyperparameter Optimization](#hyperparameter-optimization)
2. [Advanced Training Features](#advanced-training-features)
3. [Production Deployment](#production-deployment)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

## Hyperparameter Optimization

### Overview

The hyperparameter optimization module uses Optuna to automatically find the best hyperparameters for your reward model training. It supports:

- **Multiple optimization algorithms**: TPE, Random, Grid Search
- **Customizable search spaces**: Learning rate, batch size, model architecture
- **Persistent storage**: Save and resume optimization runs
- **Visualization**: Interactive plots and reports
- **Early stopping**: Prune unpromising trials

### Quick Start

```bash
# Basic optimization
python src/optimize_hyperparameters.py \
    --train-data data/train.csv \
    --val-data data/val.csv \
    --n-trials 50

# With custom configuration
python src/optimize_hyperparameters.py \
    --train-data data/train.csv \
    --val-data data/val.csv \
    --base-config config/training_config.json \
    --optimization-config config/optimization_config.json \
    --n-trials 100 \
    --plot
```

### Configuration

```json
{
  "output_dir": "./hyperparameter_optimization",
  "optimize_learning_rate": true,
  "optimize_batch_size": true,
  "optimize_num_epochs": true,
  "optimize_weight_decay": true,
  "optimize_warmup_steps": true,
  "optimize_gradient_accumulation": true,
  "optimize_model_architecture": true,
  "optimize_data_validation": true,
  "n_trials": 50,
  "timeout": 3600,
  "early_stopping_patience": 10
}
```

### Programmatic Usage

```python
from src.hyperparameter_optimization import HyperparameterOptimizer, create_optimization_config
from src.train_reward_model import create_default_config

# Load configurations
base_config = create_default_config()
optimization_config = create_optimization_config()

# Create optimizer
optimizer = HyperparameterOptimizer(base_config, optimization_config)

# Run optimization
results = optimizer.optimize(
    train_data_path="data/train.csv",
    val_data_path="data/val.csv",
    n_trials=50
)

# Get best configuration
best_config = optimizer.get_best_config()

# Generate plots
optimizer.plot_optimization_history()
```

### Optimization Results

The optimizer generates several output files:

- `best_parameters.json`: Best hyperparameters found
- `optimization_report.md`: Detailed optimization report
- `optimization_plots.png`: Visualization plots
- `optimization_study.pkl`: Optuna study object

## Advanced Training Features

### Overview

The advanced training module provides modern training capabilities:

- **Multiple optimizers**: AdamW, Lion, AdaFactor
- **Learning rate schedulers**: Cosine, Step, OneCycle, Linear
- **Mixed precision training**: FP16/BF16 for faster training
- **Gradient checkpointing**: Memory-efficient training
- **Curriculum learning**: Gradually increase difficulty
- **Label smoothing**: Better generalization
- **Advanced regularization**: Dropout, weight decay

### Quick Start

```python
from src.advanced_trainer import AdvancedRewardTrainer, create_advanced_training_config

# Load configuration
config = create_advanced_training_config()

# Create trainer
trainer = AdvancedRewardTrainer(config)

# Setup model and data
trainer.model = your_model
trainer.tokenizer = your_tokenizer
trainer.train_dataloader = train_dataloader
trainer.val_dataloader = val_dataloader

# Create optimizer and scheduler
trainer.optimizer = trainer.create_optimizer(trainer.model)
trainer.scheduler = trainer.create_scheduler(
    trainer.optimizer, 
    num_training_steps=len(train_dataloader) * config["num_epochs"]
)

# Train
history = trainer.train(num_epochs=config["num_epochs"])
```

### Configuration

```json
{
  "optimizer": "adamw",
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "scheduler": "cosine",
  "warmup_steps": 100,
  "use_mixed_precision": true,
  "use_gradient_checkpointing": true,
  "use_curriculum_learning": false,
  "use_label_smoothing": true,
  "label_smoothing_factor": 0.1,
  "batch_size": 4,
  "num_epochs": 3
}
```

### Optimizer Options

#### AdamW (Recommended)
```python
{
  "optimizer": "adamw",
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "betas": [0.9, 0.999],
  "eps": 1e-8
}
```

#### Lion (Alternative)
```python
{
  "optimizer": "lion",
  "learning_rate": 1e-4,
  "weight_decay": 0.01
}
```

#### AdaFactor (Memory Efficient)
```python
{
  "optimizer": "adafactor",
  "learning_rate": 1e-3,
  "weight_decay": 0.01,
  "clip_threshold": 1.0,
  "decay_rate": -0.8
}
```

### Scheduler Options

#### Cosine Annealing
```python
{
  "scheduler": "cosine",
  "warmup_steps": 100
}
```

#### OneCycle
```python
{
  "scheduler": "onecycle",
  "warmup_ratio": 0.1
}
```

#### Step
```python
{
  "scheduler": "step",
  "step_size": 1000,
  "gamma": 0.1
}
```

## Production Deployment

### Overview

The production deployment guide covers:

- **Docker containerization**: Easy deployment and scaling
- **FastAPI serving**: High-performance API endpoints
- **Kubernetes deployment**: Orchestration and scaling
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **A/B testing**: Model comparison framework
- **Security**: Authentication and input validation

### Quick Start

#### 1. Build Docker Image

```bash
# Build image
docker build -t betterai/reward-model-api:latest .

# Run container
docker run -p 8000:8000 betterai/reward-model-api:latest
```

#### 2. Deploy with Docker Compose

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps
```

#### 3. Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

### Monitoring

#### Prometheus Metrics

The API automatically exposes metrics at `/metrics`:

- `reward_model_requests_total`: Request count by endpoint
- `reward_model_request_duration_seconds`: Request duration
- `reward_model_prediction_duration_seconds`: Prediction time
- `reward_score_distribution`: Score distribution

#### Grafana Dashboard

Access the dashboard at `http://localhost:3000` (admin/admin):

- Request rate and response time
- Model performance metrics
- System resource usage
- Error rates and alerts

### A/B Testing

```python
from src.ab_testing.ab_test_manager import ABTestManager
import redis

# Initialize
redis_client = redis.Redis(host='localhost', port=6379)
ab_manager = ABTestManager(redis_client)

# Create experiment
ab_manager.create_experiment(
    experiment_id="model_comparison",
    models={
        "model_a": "path/to/model_a",
        "model_b": "path/to/model_b"
    },
    traffic_split={"model_a": 0.5, "model_b": 0.5}
)

# Get model for request
model_path = ab_manager.get_model_for_request(
    experiment_id="model_comparison",
    user_id="user123"
)

# Record metrics
ab_manager.record_metric(
    experiment_id="model_comparison",
    model_name="model_a",
    metric_name="accuracy",
    value=0.85
)
```

## Usage Examples

### Complete Training Pipeline

```python
# 1. Data validation
from src.data_validator import DataValidator

validator = DataValidator(config)
cleaned_data, results = validator.validate_dataset(raw_data)

# 2. Hyperparameter optimization
from src.hyperparameter_optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(base_config, opt_config)
results = optimizer.optimize(train_data, val_data, n_trials=50)
best_config = optimizer.get_best_config()

# 3. Advanced training
from src.advanced_trainer import AdvancedRewardTrainer

trainer = AdvancedRewardTrainer(best_config)
history = trainer.train(num_epochs=3)

# 4. Evaluation
from src.evaluation_metrics import RewardModelEvaluator

evaluator = RewardModelEvaluator(model, tokenizer)
report = evaluator.generate_evaluation_report(test_data)

# 5. Deploy
# Use the production deployment guide
```

### CLI Usage

```bash
# Validate data
betterai validate --input data.csv --output cleaned.csv

# Optimize hyperparameters
python src/optimize_hyperparameters.py \
    --train-data train.csv \
    --val-data val.csv \
    --n-trials 50 \
    --plot

# Train with advanced features
betterai train --config config/advanced_training_config.json

# Evaluate model
betterai evaluate --model-path ./models/reward_model --test-data test.csv

# Deploy
docker-compose up -d
```

## Best Practices

### Hyperparameter Optimization

1. **Start with reasonable bounds**: Don't search the entire parameter space
2. **Use early stopping**: Prune unpromising trials early
3. **Monitor optimization**: Check plots and reports regularly
4. **Validate results**: Test best parameters on holdout set
5. **Persistent storage**: Save optimization state for resuming

### Advanced Training

1. **Mixed precision**: Use FP16/BF16 for faster training
2. **Gradient checkpointing**: Enable for memory efficiency
3. **Learning rate scheduling**: Use cosine annealing for best results
4. **Regularization**: Apply label smoothing and weight decay
5. **Curriculum learning**: Gradually increase difficulty

### Production Deployment

1. **Health checks**: Implement comprehensive health monitoring
2. **Rate limiting**: Protect against abuse
3. **Caching**: Cache predictions for performance
4. **Monitoring**: Set up alerts for anomalies
5. **Security**: Validate inputs and authenticate requests
6. **Scaling**: Use horizontal scaling for high load
7. **Backup**: Regular model and data backups

### Performance Optimization

1. **Model optimization**: Use TorchScript and quantization
2. **Batch processing**: Process multiple requests together
3. **Async processing**: Use async/await for I/O operations
4. **Connection pooling**: Reuse database connections
5. **CDN**: Use CDN for static assets

### Monitoring and Alerting

1. **Key metrics**: Track request rate, response time, error rate
2. **Business metrics**: Monitor reward score distribution
3. **System metrics**: CPU, memory, disk usage
4. **Alerts**: Set up alerts for anomalies
5. **Dashboards**: Create comprehensive dashboards

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, enable gradient checkpointing
2. **Slow Training**: Enable mixed precision, use better optimizer
3. **Poor Performance**: Run hyperparameter optimization
4. **API Errors**: Check input validation, monitor logs
5. **Deployment Issues**: Verify Docker/Kubernetes configuration

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile model inference
import cProfile
cProfile.run('model.predict(text)')

# Check memory usage
import torch
torch.cuda.memory_summary()
```

This advanced features guide provides comprehensive coverage of the new capabilities added to BetterAI. These features enable more efficient training, better model performance, and production-ready deployment. 