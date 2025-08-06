"""
Advanced training features for reward models.

This module provides advanced training capabilities including:
- Learning rate schedulers (cosine, step, warmup)
- Multiple optimizers (AdamW, Lion, AdaFactor)
- Mixed precision training
- Gradient checkpointing
- Curriculum learning
- Regularization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ExponentialLR, 
    ReduceLROnPlateau,
    OneCycleLR
)
from torch.cuda.amp import GradScaler, autocast
import logging
import math
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import json
import numpy as np
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AdvancedRewardTrainer:
    """Advanced trainer with modern training features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Advanced features
        self.use_mixed_precision = config.get("use_mixed_precision", False)
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
        self.use_curriculum_learning = config.get("use_curriculum_learning", False)
        self.use_label_smoothing = config.get("use_label_smoothing", False)
        self.label_smoothing_factor = config.get("label_smoothing_factor", 0.1)
        
        # Setup advanced features
        self._setup_mixed_precision()
        self._setup_gradient_checkpointing()
        
        logger.info(f"🚀 Initialized advanced trainer with device: {self.device}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.use_mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("ℹ️ Mixed precision training disabled")
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing."""
        if self.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.info("ℹ️ Gradient checkpointing disabled")
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer with advanced options.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimizer instance
        """
        optimizer_type = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 2e-5)
        weight_decay = self.config.get("weight_decay", 0.01)
        
        # Get parameters to optimize
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == "lion":
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type.lower() == "adafactor":
            optimizer = optim.Adafactor(
                optimizer_grouped_parameters,
                lr=learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        logger.info(f"✅ Created {optimizer_type} optimizer with lr={learning_rate}")
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int) -> Any:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            num_training_steps: Total number of training steps
            
        Returns:
            Scheduler instance
        """
        scheduler_type = self.config.get("scheduler", "linear")
        warmup_steps = self.config.get("warmup_steps", 0)
        warmup_ratio = self.config.get("warmup_ratio", 0.1)
        
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_type.lower() == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type.lower() == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=0
            )
        elif scheduler_type.lower() == "step":
            step_size = self.config.get("step_size", num_training_steps // 10)
            scheduler = StepLR(
                optimizer,
                step_size=step_size,
                gamma=self.config.get("gamma", 0.1)
            )
        elif scheduler_type.lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.config.get("gamma", 0.95)
            )
        elif scheduler_type.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.get("learning_rate", 2e-5),
                total_steps=num_training_steps,
                pct_start=warmup_steps / num_training_steps
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        logger.info(f"✅ Created {scheduler_type} scheduler with {warmup_steps} warmup steps")
        return scheduler
    
    def create_loss_function(self) -> Callable:
        """Create loss function with optional label smoothing."""
        if self.use_label_smoothing:
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing_factor)
        else:
            return nn.CrossEntropyLoss()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
        
        return {"val_loss": loss.item()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = []
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Apply curriculum learning if enabled
            if self.use_curriculum_learning:
                batch = self._apply_curriculum_learning(batch)
            
            # Training step
            step_metrics = self.train_step(batch)
            epoch_losses.append(step_metrics["loss"])
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{step_metrics['loss']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.global_step += 1
        
        return {"train_loss": np.mean(epoch_losses)}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        val_losses = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                step_metrics = self.validation_step(batch)
                val_losses.append(step_metrics["val_loss"])
        
        return {"val_loss": np.mean(val_losses)}
    
    def _apply_curriculum_learning(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply curriculum learning to the batch."""
        # Simple curriculum: gradually increase sequence length
        max_length = self.config.get("max_length", 512)
        curriculum_factor = min(1.0, self.global_step / 1000)  # Gradually increase over 1000 steps
        
        current_max_length = int(max_length * curriculum_factor)
        
        # Truncate sequences if needed
        if "input_ids" in batch:
            batch["input_ids"] = batch["input_ids"][:, :current_max_length]
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"][:, :current_max_length]
        
        return batch
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics["train_loss"])
            
            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.validate_epoch()
                val_losses.append(val_metrics["val_loss"])
                
                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model")
                
                logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}, "
                          f"Val Loss = {val_metrics['val_loss']:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every", 1) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses
        }
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get("output_dir", "./models"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"📂 Loaded checkpoint: {checkpoint_path}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def set_learning_rate(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        logger.info(f"📈 Set learning rate to: {lr}")

def create_advanced_training_config() -> Dict[str, Any]:
    """Create a default advanced training configuration."""
    return {
        "optimizer": "adamw",  # adamw, lion, adafactor
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "scheduler": "cosine",  # linear, cosine, cosine_annealing, step, exponential, onecycle
        "warmup_steps": 100,
        "warmup_ratio": 0.1,
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "use_curriculum_learning": False,
        "use_label_smoothing": True,
        "label_smoothing_factor": 0.1,
        "save_every": 1,
        "output_dir": "./models",
        "max_length": 512,
        "batch_size": 4,
        "num_epochs": 3
    }

if __name__ == "__main__":
    # Example usage
    config = create_advanced_training_config()
    trainer = AdvancedRewardTrainer(config)
    
    # Create optimizer and scheduler
    # optimizer = trainer.create_optimizer(model)
    # scheduler = trainer.create_scheduler(optimizer, num_training_steps=1000)
    
    print("Advanced trainer initialized successfully!") 