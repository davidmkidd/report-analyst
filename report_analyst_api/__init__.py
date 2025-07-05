"""
Report Analyst API Module

This module provides FastAPI endpoints for the report analyst functionality.
Can be licensed separately from the core package.
"""

__version__ = "0.1.0"

try:
    from .main import app
    __all__ = ["app"]
except ImportError:
    # FastAPI dependencies not installed
    __all__ = [] 