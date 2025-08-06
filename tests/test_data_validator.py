#!/usr/bin/env python3
"""
Unit tests for the data validation module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import json
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_validator import DataValidator

class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
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
        
        self.validator = DataValidator(self.config)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            "chosen": [
                "This is a good response that should be chosen.",
                "Another good response with more content.",
                "A very short response",  # Too short
                "This response is identical to the rejected one.",  # Identical
                "This response has <html>tags</html> in it.",  # HTML
                "This response has a URL: https://example.com",  # URL
                "This is a very long response " * 100,  # Too long
                "This is a normal response with reasonable length and content."
            ],
            "rejected": [
                "This is a bad response that should be rejected.",
                "Another bad response with less content.",
                "A very short response",  # Too short
                "This response is identical to the rejected one.",  # Identical
                "This response has <html>tags</html> in it.",  # HTML
                "This response has a URL: https://example.com",  # URL
                "This is a very long response " * 100,  # Too long
                "This is a normal response with reasonable length and content."
            ]
        })
    
    def test_validate_structure(self):
        """Test structure validation."""
        # Test with valid data
        result = self.validator._validate_structure(self.sample_data)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Test with missing columns
        invalid_data = pd.DataFrame({"chosen": ["test"]})
        with self.assertRaises(ValueError):
            self.validator._validate_structure(invalid_data)
        
        # Test with empty dataframe
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.validator._validate_structure(empty_data)
    
    def test_validate_data_types(self):
        """Test data type validation."""
        # Test with valid data
        result = self.validator._validate_data_types(self.sample_data)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Test with non-string data
        mixed_data = pd.DataFrame({
            "chosen": ["text", 123, None, ""],
            "rejected": ["text", "text", "text", "text"]
        })
        result = self.validator._validate_data_types(mixed_data)
        # Should remove rows with empty strings
        self.assertLess(len(result), len(mixed_data))
    
    def test_validate_content(self):
        """Test content validation."""
        result = self.validator._validate_content(self.sample_data)
        
        # Should remove rows that are too short, too long, identical, or have HTML
        self.assertLess(len(result), len(self.sample_data))
        
        # Check that remaining data doesn't have HTML tags
        for _, row in result.iterrows():
            self.assertNotIn("<html>", row["chosen"])
            self.assertNotIn("<html>", row["rejected"])
    
    def test_filter_by_quality(self):
        """Test quality filtering."""
        # Add quality scores to test data
        test_data = self.sample_data.copy()
        test_data["quality_score"] = [0.8, 0.6, 0.3, 0.9, 0.7, 0.5, 0.2, 0.8]
        
        # Test with different quality thresholds
        self.validator.min_quality_score = 0.5
        result = self.validator._filter_by_quality(test_data)
        
        # Should filter out low quality scores
        self.assertLess(len(result), len(test_data))
        
        # Check that remaining scores are above threshold
        for _, row in result.iterrows():
            self.assertGreaterEqual(row["quality_score"], 0.5)
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        # Create data with duplicates
        duplicate_data = pd.DataFrame({
            "chosen": ["A", "A", "B", "C", "B"],
            "rejected": ["X", "X", "Y", "Z", "Y"]
        })
        
        result = self.validator._remove_duplicates(duplicate_data)
        
        # Should remove duplicates
        self.assertLess(len(result), len(duplicate_data))
        
        # Check that no duplicates remain
        chosen_values = result["chosen"].tolist()
        rejected_values = result["rejected"].tolist()
        
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                self.assertFalse(
                    chosen_values[i] == chosen_values[j] and 
                    rejected_values[i] == rejected_values[j]
                )
    
    def test_validate_balance(self):
        """Test balance validation."""
        result = self.validator._validate_balance(self.sample_data)
        
        # Should return the same data
        self.assertEqual(len(result), len(self.sample_data))
        
        # Check that balance stats are calculated
        self.assertIn("validation_steps", self.validator.validation_results)
        self.assertIn("balance", self.validator.validation_results["validation_steps"])
    
    def test_final_cleaning(self):
        """Test final cleaning."""
        # Add some test data with extra whitespace
        test_data = pd.DataFrame({
            "chosen": ["  Text with extra spaces  ", "Normal text", "  "],
            "rejected": ["  More text  ", "More normal text", "  "]
        })
        
        result = self.validator._final_cleaning(test_data)
        
        # Should clean whitespace and remove empty strings
        self.assertLess(len(result), len(test_data))
        
        # Check that remaining text is cleaned
        for _, row in result.iterrows():
            self.assertEqual(row["chosen"], row["chosen"].strip())
            self.assertEqual(row["rejected"], row["rejected"].strip())
    
    def test_clean_text(self):
        """Test text cleaning function."""
        # Test HTML removal
        html_text = "Text with <html>tags</html> and <p>more</p>"
        cleaned = self.validator._clean_text(html_text)
        self.assertNotIn("<html>", cleaned)
        self.assertNotIn("<p>", cleaned)
        
        # Test whitespace normalization
        whitespace_text = "  Text   with    extra    spaces  "
        cleaned = self.validator._clean_text(whitespace_text)
        self.assertEqual(cleaned, "Text with extra spaces")
        
        # Test character normalization
        char_text = "Text with "smart" quotes and – dashes"
        cleaned = self.validator._clean_text(char_text)
        self.assertIn('"', cleaned)
        self.assertIn('-', cleaned)
        
        # Test excessive punctuation
        punct_text = "Text with!!! excessive??? punctuation..."
        cleaned = self.validator._clean_text(punct_text)
        self.assertNotIn("!!!", cleaned)
        self.assertNotIn("???", cleaned)
        self.assertNotIn("...", cleaned)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        test_data = pd.DataFrame({
            "chosen": ["A reasonable length response with good content"],
            "rejected": ["Another reasonable response"],
            "similarity": [0.3],
            "has_html": [False],
            "has_urls": [False]
        })
        
        scores = self.validator._calculate_quality_score(test_data)
        
        # Should return scores between 0 and 1
        self.assertEqual(len(scores), 1)
        self.assertGreaterEqual(scores.iloc[0], 0)
        self.assertLessEqual(scores.iloc[0], 1)
    
    def test_complete_validation_pipeline(self):
        """Test the complete validation pipeline."""
        result, validation_results = self.validator.validate_dataset(self.sample_data)
        
        # Should return cleaned data and results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(validation_results, dict)
        
        # Should have validation steps
        self.assertIn("validation_steps", validation_results)
        self.assertIn("original_count", validation_results)
        self.assertIn("final_count", validation_results)
        self.assertIn("removed_count", validation_results)
        self.assertIn("retention_rate", validation_results)
        
        # Final count should be less than or equal to original
        self.assertLessEqual(validation_results["final_count"], validation_results["original_count"])
        
        # Retention rate should be between 0 and 1
        self.assertGreaterEqual(validation_results["retention_rate"], 0)
        self.assertLessEqual(validation_results["retention_rate"], 1)
    
    def test_save_validation_report(self):
        """Test saving validation report."""
        # Run validation
        result, validation_results = self.validator.validate_dataset(self.sample_data)
        
        # Save report to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name
        
        try:
            self.validator.save_validation_report(report_path)
            
            # Check that file exists and contains valid JSON
            self.assertTrue(os.path.exists(report_path))
            
            with open(report_path, 'r') as f:
                saved_data = json.load(f)
            
            # Check that saved data matches validation results
            self.assertEqual(saved_data["original_count"], validation_results["original_count"])
            self.assertEqual(saved_data["final_count"], validation_results["final_count"])
            
        finally:
            # Clean up
            if os.path.exists(report_path):
                os.unlink(report_path)
    
    def test_print_summary(self):
        """Test summary printing."""
        # Run validation
        result, validation_results = self.validator.validate_dataset(self.sample_data)
        
        # This should not raise an exception
        try:
            self.validator.print_summary()
        except Exception as e:
            self.fail(f"print_summary() raised {e} unexpectedly!")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very large dataset
        large_data = pd.DataFrame({
            "chosen": ["Test response"] * 10000,
            "rejected": ["Test response"] * 10000
        })
        
        result, validation_results = self.validator.validate_dataset(large_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(validation_results, dict)
        
        # Test with all identical data
        identical_data = pd.DataFrame({
            "chosen": ["Same text"] * 10,
            "rejected": ["Same text"] * 10
        })
        
        result, validation_results = self.validator.validate_dataset(identical_data)
        # Should remove duplicates
        self.assertLess(len(result), len(identical_data))
        
        # Test with all empty strings
        empty_data = pd.DataFrame({
            "chosen": [""] * 5,
            "rejected": [""] * 5
        })
        
        result, validation_results = self.validator.validate_dataset(empty_data)
        # Should remove empty strings
        self.assertEqual(len(result), 0)

if __name__ == "__main__":
    unittest.main() 