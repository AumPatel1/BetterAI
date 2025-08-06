#!/usr/bin/env python3
"""
Test runner for BetterAI project.

This script runs all unit tests and provides a summary of results.
"""

import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests in the project."""
    # Add the src directory to the path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code) 