# Data Validation and Cleaning

This document describes the comprehensive data validation and cleaning features for reward model training data.

## Overview

The data validation system ensures high-quality training data by performing multiple validation steps:

1. **Structure Validation** - Checks for required columns and basic data integrity
2. **Data Type Validation** - Ensures proper data types and handles missing values
3. **Content Validation** - Validates text quality, length, and similarity
4. **Quality Filtering** - Applies quality scoring and filtering
5. **Duplicate Removal** - Removes exact and near-duplicate samples
6. **Balance Validation** - Analyzes dataset balance and characteristics
7. **Final Cleaning** - Performs text normalization and final preparation

## Features

### 🔍 Comprehensive Validation
- **Length constraints** for chosen and rejected responses
- **Similarity detection** to identify too-similar pairs
- **Content filtering** for HTML, URLs, and special characters
- **Quality scoring** based on multiple metrics
- **Duplicate detection** using hashing and similarity

### 🧹 Text Cleaning
- **Whitespace normalization** and trimming
- **HTML tag removal** and text extraction
- **Character normalization** (quotes, dashes, punctuation)
- **Excessive punctuation** removal

### 📊 Quality Metrics
- **Length-based scoring** (preferring reasonable lengths)
- **Diversity scoring** (lower similarity is better)
- **Content quality scoring** (no HTML, URLs, etc.)
- **Combined quality score** for filtering

### 📈 Reporting and Analysis
- **Detailed validation reports** in JSON format
- **Statistics and metrics** for each validation step
- **Summary printing** with key statistics
- **Cleaned data saving** for further use

## Usage

### Integrated with Training

The data validation is automatically integrated into the training pipeline. Enable it in your training configuration:

```json
{
  "data_validation": {
    "enabled": true,
    "min_length": 10,
    "max_length": 2000,
    "min_chosen_length": 20,
    "max_chosen_length": 1500,
    "min_rejected_length": 20,
    "max_rejected_length": 1500,
    "max_similarity": 0.90,
    "min_quality_score": 0.3,
    "remove_html": true,
    "remove_urls": false,
    "save_cleaned_data": true,
    "save_validation_report": true
  }
}
```

### Standalone Validation

Use the standalone validation script for data preparation:

```bash
# Basic validation
python src/validate_data.py --input data.csv --output cleaned_data.csv

# With custom configuration
python src/validate_data.py --input data.csv --output cleaned_data.csv --config config/validation_config.json

# With custom parameters
python src/validate_data.py --input data.csv --output cleaned_data.csv \
    --min-length 20 --max-length 1000 --max-similarity 0.85
```

### Programmatic Usage

```python
from data_validator import DataValidator

# Create validator with configuration
config = {
    "min_length": 10,
    "max_length": 2000,
    "max_similarity": 0.95,
    "min_quality_score": 0.0,
    "remove_html": True
}

validator = DataValidator(config)

# Validate dataset
import pandas as pd
df = pd.read_csv("data.csv")
cleaned_df, validation_results = validator.validate_dataset(df)

# Save results
validator.save_validation_report("validation_report.json")
validator.print_summary()
```

## Configuration Parameters

### Length Constraints
- `min_length`: Minimum text length (default: 10)
- `max_length`: Maximum text length (default: 2000)
- `min_chosen_length`: Minimum chosen response length (default: 10)
- `max_chosen_length`: Maximum chosen response length (default: 2000)
- `min_rejected_length`: Minimum rejected response length (default: 10)
- `max_rejected_length`: Maximum rejected response length (default: 2000)

### Quality Filters
- `max_similarity`: Maximum similarity between chosen and rejected (default: 0.95)
- `min_quality_score`: Minimum quality score threshold (default: 0.0)

### Content Filters
- `remove_html`: Remove samples with HTML content (default: True)
- `remove_urls`: Remove samples with URL content (default: False)

### Output Options
- `save_cleaned_data`: Save cleaned dataset (default: True)
- `save_validation_report`: Save validation report (default: True)

## Validation Steps

### 1. Structure Validation
- Checks for required columns (`chosen`, `rejected`)
- Validates dataset is not empty
- Removes rows with missing values

### 2. Data Type Validation
- Converts columns to string type
- Removes empty strings and whitespace-only entries

### 3. Content Validation
- **Length filtering**: Removes samples outside length constraints
- **Identical detection**: Removes samples where chosen == rejected
- **Similarity detection**: Removes samples with high similarity
- **Content filtering**: Removes HTML, URLs if configured

### 4. Quality Filtering
- **Quality scoring**: Calculates quality score for each sample
- **Score filtering**: Removes samples below quality threshold
- **Length ratio**: Removes samples with extreme length differences

### 5. Duplicate Removal
- **Exact duplicates**: Removes identical sample pairs
- **Chosen duplicates**: Removes samples with identical chosen responses
- **Rejected duplicates**: Removes samples with identical rejected responses

### 6. Balance Validation
- **Length analysis**: Analyzes length distributions
- **Correlation analysis**: Checks length correlation between chosen/rejected
- **Advantage analysis**: Checks if chosen responses are generally better

### 7. Final Cleaning
- **Text normalization**: Cleans and normalizes text content
- **Final filtering**: Removes any remaining empty strings
- **Column selection**: Keeps only essential columns

## Quality Scoring

The quality score is calculated based on multiple factors:

1. **Length scoring** (0.6 points max):
   - 0.3 points for chosen response in optimal range (50-500 chars)
   - 0.3 points for rejected response in optimal range (50-500 chars)

2. **Diversity scoring** (0.4 points max):
   - Higher score for lower similarity between chosen and rejected

3. **Content quality** (0.2 points max):
   - 0.1 points for no HTML content
   - 0.1 points for no URL content

## Output Files

### Validation Report (`validation_report.json`)
```json
{
  "original_count": 10000,
  "final_count": 8500,
  "removed_count": 1500,
  "retention_rate": 0.85,
  "validation_steps": {
    "structure": { ... },
    "data_types": { ... },
    "content": { ... },
    "quality": { ... },
    "duplicates": { ... },
    "balance": { ... }
  }
}
```

### Cleaned Data (`cleaned_data.csv`)
- Contains only the `chosen` and `rejected` columns
- All samples have passed validation
- Text has been cleaned and normalized

## Best Practices

### 1. Start Conservative
Begin with lenient validation parameters and gradually tighten them:
```json
{
  "min_quality_score": 0.0,
  "max_similarity": 0.98,
  "min_length": 5
}
```

### 2. Monitor Retention Rate
Aim for 70-90% retention rate. If too low, adjust parameters:
```bash
python src/validate_data.py --input data.csv --output cleaned.csv --min-quality-score 0.1
```

### 3. Check Validation Report
Always review the validation report to understand what was removed:
```python
with open("validation_report.json", "r") as f:
    report = json.load(f)
print(f"Removed {report['removed_count']} samples ({report['retention_rate']:.1%} retention)")
```

### 4. Iterate and Refine
Use the validation results to improve your data collection process:
- If many duplicates are found, improve data collection
- If quality scores are low, review annotation guidelines
- If length distributions are poor, adjust collection criteria

## Troubleshooting

### Common Issues

1. **Low retention rate**: Reduce quality thresholds or length constraints
2. **High similarity scores**: Lower `max_similarity` threshold
3. **Missing columns**: Ensure your CSV has `chosen` and `rejected` columns
4. **Memory issues**: Process data in chunks for very large datasets

### Performance Tips

1. **Large datasets**: Use the standalone script for datasets > 100K samples
2. **Memory optimization**: Process in batches if memory is limited
3. **Parallel processing**: Consider parallel validation for very large datasets

## Examples

### Example 1: Basic Validation
```bash
python src/validate_data.py --input raw_data.csv --output clean_data.csv
```

### Example 2: Strict Quality Filtering
```bash
python src/validate_data.py --input raw_data.csv --output clean_data.csv \
    --min-quality-score 0.5 --max-similarity 0.8 --min-length 50
```

### Example 3: Custom Configuration
```bash
python src/validate_data.py --input raw_data.csv --output clean_data.csv \
    --config config/strict_validation.json
```

### Example 4: Training Integration
```python
# In your training script
config = {
    "data_validation": {
        "enabled": True,
        "min_quality_score": 0.3,
        "max_similarity": 0.9,
        "save_cleaned_data": True
    }
}

trainer = RewardModelTrainer(config)
# Validation happens automatically during data loading
``` 