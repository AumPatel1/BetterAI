import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from trl import RewardTrainer, RewardConfig

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

def main():
    """Train a reward model using pairwise preferences dataset."""
    
    print("🚀 Starting reward model training...")
    
    # 1. Import Libraries (already done at top)
    # 2. Define Model Name
    model_name = "roberta-base"
    print(f"📋 Using model: {model_name}")
    
    # 3. Load Tokenizer and Model
    print("🔧 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1  # Single reward score
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set max length to prevent sequence length issues
    max_length = 512
    tokenizer.model_max_length = max_length
    
    # Create custom data collator for handling variable-length sequences
    data_collator = CustomRewardDataCollator(tokenizer=tokenizer, max_length=max_length)
    
    print(f"✅ Model loaded with {model.num_parameters():,} parameters")
    
    # 4. Load and Prepare Dataset
    print("📊 Loading pairwise preferences dataset...")
    dataset = load_dataset('csv', data_files='data/pairwise_preferences_advanced.csv')
    print(f"📈 Dataset loaded: {dataset}")
    
    def preprocess_function(examples):
        """Preprocess the dataset for reward training."""
        # Tokenize the chosen and rejected responses
        chosen_texts = examples["chosen"]
        rejected_texts = examples["rejected"]
        
        # Tokenize with truncation
        chosen_tokens = tokenizer(
            chosen_texts,
            truncation=True,
            padding=False,  # Don't pad here, let the collator handle it
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        rejected_tokens = tokenizer(
            rejected_texts,
            truncation=True,
            padding=False,  # Don't pad here, let the collator handle it
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"]
        }
    
    print("🔄 Preprocessing dataset...")
    processed_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    print(f"✅ Dataset preprocessed: {len(processed_dataset)} samples")
    
    # 5. Define Training Arguments
    print("⚙️ Setting up training arguments...")
    training_args = RewardConfig(
        output_dir='./models/eval2reward_model_advanced',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=1,
        remove_unused_columns=False,
        warmup_steps=10,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=None,  # Disable wandb/tensorboard logging
        bf16=False  # Disable bf16 training
    )
    
    # 6. Initialize the RewardTrainer
    print("🎯 Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
    # 7. Train the Model
    print("🏋️ Starting training...")
    trainer.train()
    
    # 8. Save the Final Model
    print("💾 Saving final model...")
    trainer.save_model()
    
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved to: ./models/eval2reward_model_advanced")
    print(f"📊 Training stats:")
    print(f"   - Total steps: {trainer.state.global_step}")
    if trainer.state.log_history and 'loss' in trainer.state.log_history[-1]:
        print(f"   - Training loss: {trainer.state.log_history[-1]['loss']:.4f}")
    else:
        print(f"   - Training loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")

if __name__ == "__main__":
    main() 