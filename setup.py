#!/usr/bin/env python3
"""
Setup script for BetterAI package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="betterai",
    version="0.1.0",
    author="BetterAI Team",
    author_email="contact@betterai.com",
    description="Advanced Reward Model Training & Evaluation Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BetterAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "betterai=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "betterai": [
            "config/*.json",
            "docs/*.md",
        ],
    },
    keywords="machine-learning, reward-modeling, transformers, evaluation, bias-detection",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/BetterAI/issues",
        "Source": "https://github.com/yourusername/BetterAI",
        "Documentation": "https://betterai.readthedocs.io/",
    },
) 