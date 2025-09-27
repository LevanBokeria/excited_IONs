"""
excited_ions: Scientific Paper Processing Pipeline

A pipeline for fetching scientific papers, extracting metadata and experimental results,
generating summaries, and creating new experiment ideas using LLMs.
"""

__version__ = "0.1.0"
__author__ = "Levan Bokeria"

from .pipeline import Pipeline
from .config import Config

__all__ = ["Pipeline", "Config"]