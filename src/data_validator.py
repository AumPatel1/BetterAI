import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from collections import Counter
import hashlib
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and cleaning for reward model training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = {}
        self.cleaning_stats = {}
        
        # Default validation parameters
        self.min_length = config.get("min_length", 10)
        self.max_length = config.get("max_length", 2000)
        self.min_chosen_length = config.get("min_chosen_length", 10)
        self.max_chosen_length = config.get("max_chosen_length", 2000)
        self.min_rejected_length = config.get("min_rejected_length", 10)
        self.max_rejected_length = config.get("max_rejected_length", 2000)
        self.max_similarity = config.get("max_similarity", 0.95)
        self.min_quality_score = config.get("min_quality_score", 0.0)
        
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive dataset validation and cleaning."""
        logger.info("🔍 Starting comprehensive data validation...")
        
        original_count = len(df)
        self.validation_results = {
            "original_count": original_count,
            "validation_steps": {},
            "cleaning_steps": {},
            "final_count": 0,
            "removed_count": 0
        }
        
        # Step 1: Basic structure validation
        df = self._validate_structure(df)
        
        # Step 2: Data type validation
        df = self._validate_data_types(df)
        
        # Step 3: Content validation
        df = self._validate_content(df)
        
        # Step 4: Quality filtering
        df = self._filter_by_quality(df)
        
        # Step 5: Duplicate detection and removal
        df = self._remove_duplicates(df)
        
        # Step 6: Balance validation
        df = self._validate_balance(df)
        
        # Step 7: Final cleaning
        df = self._final_cleaning(df)
        
        # Calculate final statistics
        final_count = len(df)
        removed_count = original_count - final_count
        
        self.validation_results.update({
            "final_count": final_count,
            "removed_count": removed_count,
            "retention_rate": final_count / original_count if original_count > 0 else 0
        })
        
        logger.info(f"✅ Data validation completed!")
        logger.info(f"   - Original samples: {original_count}")
        logger.info(f"   - Final samples: {final_count}")
        logger.info(f"   - Removed samples: {removed_count}")
        logger.info(f"   - Retention rate: {self.validation_results['retention_rate']:.2%}")
        
        return df, self.validation_results
    
    def _validate_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate basic dataset structure."""
        logger.info("📋 Validating dataset structure...")
        
        # Check required columns
        required_columns = ["chosen", "rejected"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Remove rows with missing values in required columns
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        removed_count = initial_count - len(df)
        
        self.validation_results["validation_steps"]["structure"] = {
            "missing_columns": missing_columns,
            "empty_rows_removed": removed_count,
            "status": "passed" if not missing_columns else "failed"
        }
        
        logger.info(f"   - Removed {removed_count} rows with missing values")
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        logger.info("🔧 Validating data types...")
        
        initial_count = len(df)
        
        # Ensure chosen and rejected are strings
        for col in ["chosen", "rejected"]:
            df[col] = df[col].astype(str)
        
        # Remove rows where chosen or rejected are empty strings or whitespace only
        df = df[
            (df["chosen"].str.strip() != "") & 
            (df["rejected"].str.strip() != "")
        ]
        
        removed_count = initial_count - len(df)
        
        self.validation_results["validation_steps"]["data_types"] = {
            "converted_to_string": ["chosen", "rejected"],
            "empty_strings_removed": removed_count,
            "status": "passed"
        }
        
        logger.info(f"   - Removed {removed_count} rows with empty strings")
        return df
    
    def _validate_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate content quality and characteristics."""
        logger.info("📝 Validating content quality...")
        
        initial_count = len(df)
        issues = []
        
        # Length validation
        df["chosen_length"] = df["chosen"].str.len()
        df["rejected_length"] = df["rejected"].str.len()
        
        # Filter by length constraints
        length_mask = (
            (df["chosen_length"] >= self.min_chosen_length) &
            (df["chosen_length"] <= self.max_chosen_length) &
            (df["rejected_length"] >= self.min_rejected_length) &
            (df["rejected_length"] <= self.max_rejected_length)
        )
        
        length_removed = (~length_mask).sum()
        df = df[length_mask]
        
        if length_removed > 0:
            issues.append(f"Length constraints: {length_removed} samples removed")
        
        # Check for identical chosen and rejected responses
        identical_mask = df["chosen"] == df["rejected"]
        identical_count = identical_mask.sum()
        
        if identical_count > 0:
            issues.append(f"Identical responses: {identical_count} samples")
            df = df[~identical_mask]
        
        # Check for very similar responses (using sequence similarity)
        df["similarity"] = df.apply(
            lambda row: SequenceMatcher(None, row["chosen"], row["rejected"]).ratio(), 
            axis=1
        )
        
        too_similar_mask = df["similarity"] > self.max_similarity
        too_similar_count = too_similar_mask.sum()
        
        if too_similar_count > 0:
            issues.append(f"Too similar responses: {too_similar_count} samples")
            df = df[~too_similar_mask]
        
        # Check for common issues
        df["has_html"] = df["chosen"].str.contains(r'<[^>]+>', regex=True) | df["rejected"].str.contains(r'<[^>]+>', regex=True)
        df["has_urls"] = df["chosen"].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', regex=True) | df["rejected"].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', regex=True)
        df["has_special_chars"] = df["chosen"].str.contains(r'[^\w\s.,!?-]', regex=True) | df["rejected"].str.contains(r'[^\w\s.,!?-]', regex=True)
        
        # Remove problematic content if configured
        if self.config.get("remove_html", True):
            html_mask = df["has_html"]
            html_removed = html_mask.sum()
            if html_removed > 0:
                issues.append(f"HTML content: {html_removed} samples removed")
                df = df[~html_mask]
        
        if self.config.get("remove_urls", False):
            url_mask = df["has_urls"]
            url_removed = url_mask.sum()
            if url_removed > 0:
                issues.append(f"URL content: {url_removed} samples removed")
                df = df[~url_mask]
        
        removed_count = initial_count - len(df)
        
        self.validation_results["validation_steps"]["content"] = {
            "length_removed": length_removed,
            "identical_removed": identical_count,
            "too_similar_removed": too_similar_count,
            "issues_found": issues,
            "status": "passed"
        }
        
        logger.info(f"   - Content validation issues: {len(issues)}")
        for issue in issues:
            logger.info(f"     - {issue}")
        
        return df
    
    def _filter_by_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on quality metrics."""
        logger.info("⭐ Filtering by quality metrics...")
        
        initial_count = len(df)
        
        # Calculate quality scores
        df["quality_score"] = self._calculate_quality_score(df)
        
        # Filter by minimum quality score
        quality_mask = df["quality_score"] >= self.min_quality_score
        quality_removed = (~quality_mask).sum()
        df = df[quality_mask]
        
        # Check for extreme length differences
        df["length_ratio"] = df["chosen_length"] / (df["rejected_length"] + 1e-8)
        extreme_ratio_mask = (df["length_ratio"] < 0.1) | (df["length_ratio"] > 10)
        extreme_ratio_removed = extreme_ratio_mask.sum()
        df = df[~extreme_ratio_mask]
        
        removed_count = initial_count - len(df)
        
        self.validation_results["validation_steps"]["quality"] = {
            "quality_score_removed": quality_removed,
            "extreme_length_ratio_removed": extreme_ratio_removed,
            "avg_quality_score": df["quality_score"].mean() if len(df) > 0 else 0,
            "status": "passed"
        }
        
        logger.info(f"   - Quality filtering removed {removed_count} samples")
        logger.info(f"   - Average quality score: {df['quality_score'].mean():.3f}")
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality score for each sample."""
        scores = []
        
        for _, row in df.iterrows():
            score = 0.0
            
            # Length-based scoring
            chosen_len = len(row["chosen"])
            rejected_len = len(row["rejected"])
            
            # Prefer reasonable lengths
            if 50 <= chosen_len <= 500:
                score += 0.3
            elif 20 <= chosen_len <= 1000:
                score += 0.2
            
            if 50 <= rejected_len <= 500:
                score += 0.3
            elif 20 <= rejected_len <= 1000:
                score += 0.2
            
            # Diversity scoring (lower similarity is better)
            similarity = row.get("similarity", 0)
            score += (1 - similarity) * 0.4
            
            # Content quality scoring
            if not row.get("has_html", False):
                score += 0.1
            
            if not row.get("has_urls", False):
                score += 0.1
            
            scores.append(min(score, 1.0))
        
        return pd.Series(scores, index=df.index)
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate and near-duplicate samples."""
        logger.info("🔄 Removing duplicates...")
        
        initial_count = len(df)
        
        # Create hash for each sample
        df["sample_hash"] = df.apply(
            lambda row: hashlib.md5(f"{row['chosen']}|||{row['rejected']}".encode()).hexdigest(),
            axis=1
        )
        
        # Remove exact duplicates
        df = df.drop_duplicates(subset=["sample_hash"])
        exact_duplicates_removed = initial_count - len(df)
        
        # Remove near-duplicates based on chosen text
        df = df.drop_duplicates(subset=["chosen"], keep="first")
        chosen_duplicates_removed = initial_count - exact_duplicates_removed - len(df)
        
        # Remove near-duplicates based on rejected text
        df = df.drop_duplicates(subset=["rejected"], keep="first")
        rejected_duplicates_removed = initial_count - exact_duplicates_removed - chosen_duplicates_removed - len(df)
        
        total_removed = initial_count - len(df)
        
        self.validation_results["validation_steps"]["duplicates"] = {
            "exact_duplicates_removed": exact_duplicates_removed,
            "chosen_duplicates_removed": chosen_duplicates_removed,
            "rejected_duplicates_removed": rejected_duplicates_removed,
            "total_duplicates_removed": total_removed,
            "status": "passed"
        }
        
        logger.info(f"   - Exact duplicates removed: {exact_duplicates_removed}")
        logger.info(f"   - Chosen duplicates removed: {chosen_duplicates_removed}")
        logger.info(f"   - Rejected duplicates removed: {rejected_duplicates_removed}")
        
        return df
    
    def _validate_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and potentially balance the dataset."""
        logger.info("⚖️ Validating dataset balance...")
        
        # Analyze length distribution
        chosen_lengths = df["chosen_length"].values
        rejected_lengths = df["rejected_length"].values
        
        balance_stats = {
            "chosen_mean_length": chosen_lengths.mean(),
            "rejected_mean_length": rejected_lengths.mean(),
            "chosen_std_length": chosen_lengths.std(),
            "rejected_std_length": rejected_lengths.std(),
            "length_correlation": np.corrcoef(chosen_lengths, rejected_lengths)[0, 1] if len(chosen_lengths) > 1 else 0
        }
        
        # Check if chosen responses are generally longer (which might indicate better quality)
        length_advantage = (chosen_lengths > rejected_lengths).mean()
        balance_stats["chosen_length_advantage"] = length_advantage
        
        self.validation_results["validation_steps"]["balance"] = {
            "stats": balance_stats,
            "status": "passed"
        }
        
        logger.info(f"   - Chosen mean length: {balance_stats['chosen_mean_length']:.1f}")
        logger.info(f"   - Rejected mean length: {balance_stats['rejected_mean_length']:.1f}")
        logger.info(f"   - Chosen length advantage: {length_advantage:.2%}")
        
        return df
    
    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and preparation steps."""
        logger.info("🧹 Final cleaning...")
        
        # Clean text content
        df["chosen"] = df["chosen"].apply(self._clean_text)
        df["rejected"] = df["rejected"].apply(self._clean_text)
        
        # Remove any remaining empty strings
        df = df[
            (df["chosen"].str.strip() != "") & 
            (df["rejected"].str.strip() != "")
        ]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Keep only essential columns
        essential_columns = ["chosen", "rejected"]
        df = df[essential_columns]
        
        logger.info(f"   - Final dataset shape: {df.shape}")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text samples."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text.strip()
    
    def save_validation_report(self, output_path: str):
        """Save detailed validation report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"📊 Validation report saved to: {report_path}")
    
    def print_summary(self):
        """Print a summary of validation results."""
        print("\n" + "="*60)
        print("📊 DATA VALIDATION SUMMARY")
        print("="*60)
        
        results = self.validation_results
        
        print(f"Original samples: {results['original_count']:,}")
        print(f"Final samples: {results['final_count']:,}")
        print(f"Removed samples: {results['removed_count']:,}")
        print(f"Retention rate: {results['retention_rate']:.2%}")
        
        print("\nValidation Steps:")
        for step, details in results['validation_steps'].items():
            status = details.get('status', 'unknown')
            print(f"  {step.replace('_', ' ').title()}: {status}")
        
        print("="*60) 