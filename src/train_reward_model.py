import torch
import pandas as pd
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from transformers.data.data_collator import DataCollatorWithPadding
from trl import RewardTrainer, RewardConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from tqdm import tqdm

# Import data validation module
try:
    from data_validator import DataValidator
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DATA_VALIDATOR_AVAILABLE = False
    logger.warning("DataValidator not available. Data validation will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reward_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomRewardDataCollator(DataCollatorWithPadding):
    """Custom data collator for reward training that handles pre-tokenized data."""
    
    def __init__(self, tokenizer, max_length=512):
        super().__init__(tokenizer=tokenizer)
        self.max_length = max_length
    
    def __call__(self, features):
        # The data is already tokenized, just need to pad and convert to tensors
        chosen_input_ids = [f["input_ids_chosen"] for f in features]
        chosen_attention_mask = [f["attention_mask_chosen"] for f in features]
        rejected_input_ids = [f["input_ids_rejected"] for f in features]
        rejected_attention_mask = [f["attention_mask_rejected"] for f in features]
        
        # Pad sequences
        chosen_input_ids = self.tokenizer.pad(
            {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask},
            return_tensors="pt"
        )
        
        rejected_input_ids = self.tokenizer.pad(
            {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask},
            return_tensors="pt"
        )
        
        # Create the batch
        batch = {
            "input_ids_chosen": chosen_input_ids["input_ids"],
            "attention_mask_chosen": chosen_input_ids["attention_mask"],
            "input_ids_rejected": rejected_input_ids["input_ids"],
            "attention_mask_rejected": rejected_input_ids["attention_mask"]
        }
        
        return batch

class RewardModelTrainer:
    """Enhanced reward model trainer with better error handling and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Create output directories
        self.output_dir = Path(config.get("output_dir", "./models/reward_model"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Initializing RewardModelTrainer on device: {self.device}")
    
    def load_tokenizer_and_model(self):
        """Load and configure tokenizer and model."""
        try:
            model_name = self.config["model_name"]
            logger.info(f"📋 Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=False,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set max length
            max_length = self.config.get("max_length", 512)
            self.tokenizer.model_max_length = max_length
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=1,  # Single reward score
                torch_dtype=torch.float16 if self.config.get("use_fp16", False) else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
            logger.info(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with optional validation."""
        try:
            data_path = self.config["data_path"]
            logger.info(f"📊 Loading dataset from: {data_path}")
            
            # Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found: {data_path}")
            
            # Load dataset as pandas DataFrame for validation
            df = pd.read_csv(data_path)
            logger.info(f"📈 Dataset loaded: {df.shape}")
            
            # Data validation if enabled
            if (self.config.get("data_validation", {}).get("enabled", False) and 
                DATA_VALIDATOR_AVAILABLE):
                
                logger.info("🔍 Running data validation and cleaning...")
                validation_config = self.config.get("data_validation", {})
                
                validator = DataValidator(validation_config)
                df, validation_results = validator.validate_dataset(df)
                
                # Save validation report if requested
                if validation_config.get("save_validation_report", True):
                    report_path = self.output_dir / "validation_report.json"
                    validator.save_validation_report(str(report_path))
                
                # Save cleaned data if requested
                if validation_config.get("save_cleaned_data", True):
                    cleaned_data_path = self.output_dir / "cleaned_data.csv"
                    df.to_csv(cleaned_data_path, index=False)
                    logger.info(f"💾 Cleaned data saved to: {cleaned_data_path}")
                
                # Print validation summary
                validator.print_summary()
                
                # Update data path to use cleaned data
                data_path = str(cleaned_data_path)
            
            # Convert to HuggingFace dataset
            dataset = Dataset.from_pandas(df)
            logger.info(f"📈 Dataset converted: {dataset}")
            
            # Validate dataset structure
            required_columns = ["chosen", "rejected"]
            for col in required_columns:
                if col not in dataset.column_names:
                    raise ValueError(f"Missing required column: {col}")
            
            # Preprocess dataset
            max_length = self.config.get("max_length", 512)
            processed_dataset = dataset.map(
                lambda examples: self._preprocess_function(examples, max_length),
                batched=True,
                remove_columns=dataset.column_names,
                desc="Preprocessing dataset"
            )
            
            logger.info(f"✅ Dataset preprocessed: {len(processed_dataset)} samples")
            
            # Split into train/validation if specified
            if self.config.get("validation_split", 0.1) > 0:
                split_dataset = processed_dataset.train_test_split(
                    test_size=self.config["validation_split"],
                    seed=self.config.get("seed", 42)
                )
                return split_dataset
            else:
                return {"train": processed_dataset, "test": processed_dataset}
                
        except Exception as e:
            logger.error(f"❌ Error loading/preprocessing data: {str(e)}")
            raise
    
    def _preprocess_function(self, examples: Dict[str, Any], max_length: int) -> Dict[str, Any]:
        """Preprocess the dataset for reward training."""
        chosen_texts = examples["chosen"]
        rejected_texts = examples["rejected"]
        
        # Tokenize with truncation
        chosen_tokens = self.tokenizer(
            chosen_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        rejected_tokens = self.tokenizer(
            rejected_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"]
        }
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Create and configure the RewardTrainer."""
        try:
            # Create custom data collator
            data_collator = CustomRewardDataCollator(
                tokenizer=self.tokenizer, 
                max_length=self.config.get("max_length", 512)
            )
            
            # Training arguments
            training_args = RewardConfig(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config.get("num_epochs", 3),
                per_device_train_batch_size=self.config.get("batch_size", 4),
                per_device_eval_batch_size=self.config.get("eval_batch_size", 4),
                learning_rate=self.config.get("learning_rate", 2e-5),
                warmup_steps=self.config.get("warmup_steps", 100),
                weight_decay=self.config.get("weight_decay", 0.01),
                logging_steps=self.config.get("logging_steps", 10),
                save_steps=self.config.get("save_steps", 500),
                save_strategy="steps",
                save_total_limit=self.config.get("save_total_limit", 3),
                remove_unused_columns=False,
                report_to="wandb" if self.config.get("use_wandb", False) and WANDB_AVAILABLE else None,
                bf16=self.config.get("use_bf16", False),
                fp16=self.config.get("use_fp16", False),
                dataloader_num_workers=self.config.get("dataloader_num_workers", 0),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
                gradient_checkpointing=self.config.get("gradient_checkpointing", False),
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                seed=self.config.get("seed", 42),
                dataloader_pin_memory=False,
                group_by_length=True,
            )
            
            # Initialize trainer
            self.trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            logger.info("🎯 RewardTrainer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating trainer: {str(e)}")
            raise
    
    def train(self):
        """Train the reward model."""
        try:
            logger.info("🏋️ Starting training...")
            
            # Train the model
            train_result = self.trainer.train()
            
            # Save training results
            self.trainer.save_model()
            self.trainer.save_state()
            
            # Log training metrics
            logger.info("📊 Training completed!")
            logger.info(f"   - Total steps: {train_result.global_step}")
            logger.info(f"   - Training loss: {train_result.training_loss:.4f}")
            if hasattr(train_result, 'metrics') and 'train_runtime' in train_result.metrics:
                logger.info(f"   - Training time: {train_result.metrics['train_runtime']}")
            
            # Save training metrics
            metrics_path = self.output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            return train_result
            
        except Exception as e:
            logger.error(f"❌ Error during training: {str(e)}")
            raise
    
    def evaluate(self, eval_dataset: Dataset):
        """Evaluate the trained model."""
        try:
            logger.info("🔍 Evaluating model...")
            
            eval_results = self.trainer.evaluate(eval_dataset)
            
            logger.info("📊 Evaluation Results:")
            for key, value in eval_results.items():
                logger.info(f"   - {key}: {value:.4f}")
            
            # Save evaluation results
            eval_path = self.output_dir / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            return eval_results
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
            raise
    
    def save_config(self):
        """Save the training configuration."""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"💾 Configuration saved to: {config_path}")

def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for training."""
    return {
        "model_name": "roberta-base",
        "data_path": "data/pairwise_preferences_advanced.csv",
        "output_dir": "./models/reward_model_enhanced",
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
        "seed": 42
    }

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"📋 Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Configuration file not found: {config_path}. Using default config.")
        return create_default_config()
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration file: {e}")
        return create_default_config()

def main():
    """Main training function with enhanced error handling and configuration."""
    
    try:
        # Load configuration (from file or default)
        config_path = "config/training_config.json"
        if os.path.exists(config_path):
            config = load_config_from_file(config_path)
        else:
            config = create_default_config()
            logger.info("📋 Using default configuration")
        
        # Initialize wandb if enabled
        if config["use_wandb"] and WANDB_AVAILABLE:
            wandb.init(
                project="reward-model-training",
                config=config,
                name=f"reward-model-{config['model_name']}"
            )
        
        # Create trainer
        trainer = RewardModelTrainer(config)
        
        # Save configuration
        trainer.save_config()
        
        # Load model and tokenizer
        trainer.load_tokenizer_and_model()
        
        # Load and preprocess data
        datasets = trainer.load_and_preprocess_data()
        
        # Create trainer
        trainer.create_trainer(
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"] if "test" in datasets else None
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate if validation dataset exists
        if "test" in datasets:
            eval_results = trainer.evaluate(datasets["test"])
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {trainer.output_dir}")
        
        if config.get("use_wandb", False) and WANDB_AVAILABLE:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        if config.get("use_wandb", False) and WANDB_AVAILABLE:
            wandb.finish()
        raise

if __name__ == "__main__":
    main() 