"""
Report Analyst Search Backend Integration Module

This module provides integration with the search backend for PDF processing
and chunk retrieval. Can be licensed separately from the core package.
"""

__version__ = "0.1.0"

try:
    from .backend_source import SearchBackendSource
    __all__ = ["SearchBackendSource"]
except ImportError:
    # HTTP client dependencies not installed
    __all__ = [] 