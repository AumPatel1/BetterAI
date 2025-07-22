#!/usr/bin/env python3
"""
Standalone data validation script for reward model training data.

Usage:
    python validate_data.py --input data.csv --output cleaned_data.csv --config config.json
"""

import argparse
import pandas as pd
import json
import logging
from pathlib import Path
import sys
import os

# Add the current directory to the path to import data_validator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_validator import DataValidator
except ImportError as e:
    print(f"Error importing DataValidator: {e}")
    print("Make sure data_validator.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_validation_config():
    """Create default validation configuration."""
    return {
        "min_length": 10,
        "max_length": 2000,
        "min_chosen_length": 10,
        "max_chosen_length": 2000,
        "min_rejected_length": 10,
        "max_rejected_length": 2000,
        "max_similarity": 0.95,
        "min_quality_score": 0.0,
        "remove_html": True,
        "remove_urls": False
    }

def load_config(config_path):
    """Load validation configuration from file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return create_default_validation_config()

def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean reward model training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default settings
  python validate_data.py --input data.csv --output cleaned_data.csv
  
  # Validation with custom config
  python validate_data.py --input data.csv --output cleaned_data.csv --config validation_config.json
  
  # Validation with custom parameters
  python validate_data.py --input data.csv --output cleaned_data.csv --min-length 20 --max-length 1000
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file path for cleaned data"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="JSON configuration file for validation settings"
    )
    
    parser.add_argument(
        "--report", "-r",
        help="Output JSON file path for validation report"
    )
    
    # Validation parameters
    parser.add_argument(
        "--min-length",
        type=int,
        help="Minimum text length"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum text length"
    )
    
    parser.add_argument(
        "--min-chosen-length",
        type=int,
        help="Minimum chosen response length"
    )
    
    parser.add_argument(
        "--max-chosen-length",
        type=int,
        help="Maximum chosen response length"
    )
    
    parser.add_argument(
        "--min-rejected-length",
        type=int,
        help="Minimum rejected response length"
    )
    
    parser.add_argument(
        "--max-rejected-length",
        type=int,
        help="Maximum rejected response length"
    )
    
    parser.add_argument(
        "--max-similarity",
        type=float,
        help="Maximum similarity between chosen and rejected responses"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score threshold"
    )
    
    parser.add_argument(
        "--remove-html",
        action="store_true",
        help="Remove samples with HTML content"
    )
    
    parser.add_argument(
        "--remove-urls",
        action="store_true",
        help="Remove samples with URL content"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.min_length is not None:
        config["min_length"] = args.min_length
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.min_chosen_length is not None:
        config["min_chosen_length"] = args.min_chosen_length
    if args.max_chosen_length is not None:
        config["max_chosen_length"] = args.max_chosen_length
    if args.min_rejected_length is not None:
        config["min_rejected_length"] = args.min_rejected_length
    if args.max_rejected_length is not None:
        config["max_rejected_length"] = args.max_rejected_length
    if args.max_similarity is not None:
        config["max_similarity"] = args.max_similarity
    if args.min_quality_score is not None:
        config["min_quality_score"] = args.min_quality_score
    if args.remove_html:
        config["remove_html"] = True
    if args.remove_urls:
        config["remove_urls"] = True
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"📊 Loading data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"📈 Loaded {len(df)} samples")
        
        # Validate data structure
        required_columns = ["chosen", "rejected"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)
        
        # Run validation
        logger.info("🔍 Starting data validation...")
        validator = DataValidator(config)
        cleaned_df, validation_results = validator.validate_dataset(df)
        
        # Save cleaned data
        logger.info(f"💾 Saving cleaned data to: {args.output}")
        cleaned_df.to_csv(args.output, index=False)
        
        # Save validation report
        if args.report:
            logger.info(f"📊 Saving validation report to: {args.report}")
            validator.save_validation_report(args.report)
        else:
            # Save report in same directory as output
            report_path = output_path.parent / "validation_report.json"
            validator.save_validation_report(str(report_path))
        
        # Print summary
        validator.print_summary()
        
        logger.info("✅ Data validation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 