# Production Deployment Guide

This guide covers deploying BetterAI reward models in production environments with best practices for scalability, monitoring, and maintenance.

## Table of Contents

1. [Overview](#overview)
2. [Docker Containerization](#docker-containerization)
3. [FastAPI Model Serving](#fastapi-model-serving)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [A/B Testing Framework](#ab-testing-framework)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Security Considerations](#security-considerations)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Overview

Production deployment of reward models requires careful consideration of:

- **Scalability**: Handle varying load and traffic patterns
- **Reliability**: Ensure high availability and fault tolerance
- **Monitoring**: Track model performance and system health
- **Security**: Protect models and data
- **Maintenance**: Easy updates and rollbacks

## Docker Containerization

### Dockerfile

```dockerfile
# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  reward-model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/reward_model
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

## FastAPI Model Serving

### API Server

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import torch
import logging
import time
from typing import List, Dict, Any, Optional
import redis
import json
import os
from transformers import AutoTokenizer

from src.model_architectures import ModelFactory
from src.evaluation_metrics import RewardModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BetterAI Reward Model API",
    description="Production API for reward model inference",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Global model variables
model = None
tokenizer = None
evaluator = None

class RewardRequest(BaseModel):
    text: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RewardResponse(BaseModel):
    reward_score: float
    confidence: float
    processing_time: float
    model_version: str

class BatchRewardRequest(BaseModel):
    texts: List[str]
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchRewardResponse(BaseModel):
    reward_scores: List[float]
    confidences: List[float]
    processing_time: float
    model_version: str

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, tokenizer, evaluator
    
    try:
        # Load model configuration
        config_path = os.getenv("MODEL_CONFIG", "/app/config/model_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model and tokenizer
        model = ModelFactory.create_model(config)
        tokenizer = ModelFactory.get_tokenizer(config)
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Initialize evaluator
        evaluator = RewardModelEvaluator(model, tokenizer, device)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

@app.post("/predict", response_model=RewardResponse)
async def predict_reward(request: RewardRequest):
    """Predict reward score for a single text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"reward:{hash(request.text)}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            result = json.loads(cached_result)
            result["processing_time"] = time.time() - start_time
            return RewardResponse(**result)
        
        # Get reward score
        reward_score = model.get_reward_score(request.text, tokenizer)
        
        # Calculate confidence (placeholder - implement based on your needs)
        confidence = 0.95
        
        result = {
            "reward_score": reward_score,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "model_version": "1.0.0"
        }
        
        # Cache result (expire in 1 hour)
        redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return RewardResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchRewardResponse)
async def predict_reward_batch(request: BatchRewardRequest):
    """Predict reward scores for multiple texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        reward_scores = []
        confidences = []
        
        for text in request.texts:
            reward_score = model.get_reward_score(text, tokenizer)
            confidence = 0.95  # Placeholder
            
            reward_scores.append(reward_score)
            confidences.append(confidence)
        
        return BatchRewardResponse(
            reward_scores=reward_scores,
            confidences=confidences,
            processing_time=time.time() - start_time,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_texts(request: BatchRewardRequest):
    """Evaluate multiple texts with comprehensive metrics."""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not loaded")
    
    try:
        # Create evaluation dataset
        eval_data = pd.DataFrame({
            "chosen": request.texts,
            "rejected": [""] * len(request.texts)  # Placeholder
        })
        
        # Run evaluation
        results = evaluator.evaluate_preference_accuracy(eval_data)
        
        return {
            "accuracy": results.get("accuracy", 0.0),
            "precision": results.get("precision", 0.0),
            "recall": results.get("recall", 0.0),
            "f1_score": results.get("f1_score", 0.0),
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model updates
async def update_model_async(model_path: str):
    """Background task to update model."""
    global model, tokenizer
    
    try:
        # Load new model
        new_model = torch.load(model_path, map_location='cpu')
        new_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Update global variables
        model = new_model
        tokenizer = new_tokenizer
        
        logger.info("✅ Model updated successfully")
        
    except Exception as e:
        logger.error(f"❌ Model update failed: {e}")

@app.post("/model/update")
async def update_model(model_path: str, background_tasks: BackgroundTasks):
    """Update model with new version."""
    background_tasks.add_task(update_model_async, model_path)
    return {"message": "Model update started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Kubernetes Deployment

### Deployment YAML

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reward-model-api
  labels:
    app: reward-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reward-model-api
  template:
    metadata:
      labels:
        app: reward-model-api
    spec:
      containers:
      - name: reward-model-api
        image: betterai/reward-model-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/reward_model"
        - name: LOG_LEVEL
          value: "INFO"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: reward-model-service
spec:
  selector:
    app: reward-model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'reward-model-api'
    static_configs:
      - targets: ['reward-model-service:80']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-service:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Custom Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Request metrics
REQUEST_COUNT = Counter('reward_model_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('reward_model_request_duration_seconds', 'Request duration', ['endpoint'])
PREDICTION_DURATION = Histogram('reward_model_prediction_duration_seconds', 'Prediction duration')

# Model metrics
MODEL_LOAD_TIME = Gauge('reward_model_load_time_seconds', 'Model load time')
MODEL_MEMORY_USAGE = Gauge('reward_model_memory_bytes', 'Model memory usage')
MODEL_VERSION = Gauge('reward_model_version', 'Model version', ['version'])

# Business metrics
REWARD_SCORE_DISTRIBUTION = Histogram('reward_score_distribution', 'Reward score distribution', buckets=[-1.0, -0.5, 0.0, 0.5, 1.0])
CACHE_HIT_RATIO = Gauge('cache_hit_ratio', 'Cache hit ratio')

# Error metrics
PREDICTION_ERRORS = Counter('reward_model_prediction_errors_total', 'Prediction errors', ['error_type'])
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Reward Model API Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(reward_model_requests_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(reward_model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Reward Score Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(reward_score_distribution_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ]
  }
}
```

## A/B Testing Framework

```python
# src/ab_testing/ab_test_manager.py
import random
import json
import redis
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

class ABTestManager:
    """A/B testing manager for reward models."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.experiments = {}
    
    def create_experiment(self, experiment_id: str, models: Dict[str, str], 
                         traffic_split: Dict[str, float]) -> bool:
        """Create a new A/B test experiment."""
        experiment = {
            "id": experiment_id,
            "models": models,
            "traffic_split": traffic_split,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "metrics": {}
        }
        
        self.experiments[experiment_id] = experiment
        self.redis_client.set(f"experiment:{experiment_id}", json.dumps(experiment))
        
        return True
    
    def get_model_for_request(self, experiment_id: str, user_id: str) -> str:
        """Get model variant for a specific request."""
        if experiment_id not in self.experiments:
            return "default"
        
        experiment = self.experiments[experiment_id]
        
        # Use user_id to ensure consistent assignment
        random.seed(hash(user_id))
        rand_val = random.random()
        
        cumulative = 0
        for model_name, split in experiment["traffic_split"].items():
            cumulative += split
            if rand_val <= cumulative:
                return experiment["models"][model_name]
        
        return list(experiment["models"].values())[0]
    
    def record_metric(self, experiment_id: str, model_name: str, 
                     metric_name: str, value: float):
        """Record a metric for A/B testing."""
        key = f"ab_test:{experiment_id}:{model_name}:{metric_name}"
        self.redis_client.lpush(key, value)
        self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 values
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results."""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        results = {}
        
        for model_name in experiment["models"]:
            model_results = {}
            for metric in ["accuracy", "response_time", "reward_score"]:
                key = f"ab_test:{experiment_id}:{model_name}:{metric}"
                values = self.redis_client.lrange(key, 0, -1)
                if values:
                    values = [float(v) for v in values]
                    model_results[metric] = {
                        "mean": sum(values) / len(values),
                        "count": len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            results[model_name] = model_results
        
        return results
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to Docker Hub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: |
          betterai/reward-model-api:latest
          betterai/reward-model-api:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Install kubectl
      uses: azure/setup-kubectl@v1
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout restart deployment/reward-model-api
        kubectl rollout status deployment/reward-model-api
```

## Security Considerations

### API Security

```python
# src/security/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def rate_limit(requests_per_minute: int = 60):
    """Rate limiting decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Implement rate limiting logic
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Input Validation

```python
# src/security/validation.py
import re
from typing import List

def validate_text_input(text: str, max_length: int = 10000) -> bool:
    """Validate text input for security."""
    if len(text) > max_length:
        return False
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

def sanitize_text(text: str) -> str:
    """Sanitize text input."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32)
    
    return text.strip()
```

## Performance Optimization

### Model Optimization

```python
# src/optimization/model_optimization.py
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference."""
    # Enable inference mode
    model.eval()
    
    # Use TorchScript for optimization
    scripted_model = torch.jit.script(model)
    
    # Quantize model (reduce precision)
    quantized_model = quantize_dynamic(
        scripted_model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

def optimize_batch_inference(model: nn.Module, batch_size: int = 32):
    """Optimize for batch inference."""
    # Enable automatic mixed precision
    model = model.half()
    
    # Use gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model
```

### Caching Strategy

```python
# src/caching/cache_manager.py
import redis
import json
import hashlib
from typing import Any, Optional

class CacheManager:
    """Cache manager for reward model predictions."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.default_ttl = 3600  # 1 hour
    
    def get_cache_key(self, text: str, model_version: str) -> str:
        """Generate cache key."""
        content = f"{text}:{model_version}"
        return f"reward:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get(self, text: str, model_version: str) -> Optional[float]:
        """Get cached reward score."""
        key = self.get_cache_key(text, model_version)
        value = self.redis_client.get(key)
        return float(value) if value else None
    
    def set(self, text: str, model_version: str, score: float, ttl: int = None):
        """Cache reward score."""
        key = self.get_cache_key(text, model_version)
        ttl = ttl or self.default_ttl
        self.redis_client.setex(key, ttl, score)
    
    def invalidate_model_cache(self, model_version: str):
        """Invalidate all cache entries for a model version."""
        pattern = f"reward:*"
        keys = self.redis_client.keys(pattern)
        
        for key in keys:
            # Check if key corresponds to model version
            # Implementation depends on cache key structure
            pass
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model file integrity
   python -c "import torch; torch.load('model.pt')"
   
   # Verify model configuration
   python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('model_path')"
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   nvidia-smi  # For GPU
   htop        # For CPU
   
   # Reduce batch size or enable gradient checkpointing
   ```

3. **Performance Issues**
   ```bash
   # Profile model inference
   python -m cProfile -o profile.stats inference_script.py
   
   # Analyze with snakeviz
   snakeviz profile.stats
   ```

### Logging Configuration

```python
# src/logging/config.py
import logging
import logging.handlers
import os

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Special handlers for different components
    model_logger = logging.getLogger("model")
    model_logger.setLevel(logging.DEBUG)
    
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)
```

This production deployment guide provides a comprehensive framework for deploying reward models in production environments. The guide covers all essential aspects from containerization to monitoring, ensuring your models are scalable, reliable, and maintainable. 