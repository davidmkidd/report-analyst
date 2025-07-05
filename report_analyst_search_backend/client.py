"""
Search Backend HTTP Client

This module provides an HTTP client for communicating with the search backend API.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)

class SearchBackendClient:
    """HTTP client for search backend API"""
    
    def __init__(self, backend_url: str, api_key: Optional[str] = None):
        """
        Initialize the search backend client.
        
        Args:
            backend_url: Base URL of the search backend API
            api_key: Optional API key for authentication
        """
        if httpx is None:
            raise ImportError("httpx is required for search backend integration. Install with: pip install httpx")
        
        self.backend_url = backend_url.rstrip('/')
        self.api_key = api_key
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def upload_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Upload a PDF file to the search backend.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Response data including resource_id
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.post(
                    f"{self.backend_url}/upload",
                    files=files,
                    headers=headers,
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()
    
    async def get_chunks(self, resource_id: str, configuration: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get chunks for a resource from the search backend.
        
        Args:
            resource_id: ID of the resource
            configuration: Optional configuration parameters
            
        Returns:
            List of chunk data
        """
        params = {}
        if configuration:
            # Map configuration to query parameters
            if "chunk_size" in configuration:
                params["chunk_size"] = configuration["chunk_size"]
            if "chunk_overlap" in configuration:
                params["chunk_overlap"] = configuration["chunk_overlap"]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/chunks/{resource_id}",
                params=params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """
        Get the status of a resource.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Resource status information
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/resources/{resource_id}",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def delete_resource(self, resource_id: str) -> bool:
        """
        Delete a resource from the search backend.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.backend_url}/resources/{resource_id}",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False  # Resource not found, consider it deleted
            logger.error(f"Failed to delete resource {resource_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete resource {resource_id}: {e}")
            return False
    
    async def search_embeddings(self, query: str, top_k: int = 10, threshold: float = 0.0) -> Dict[str, Any]:
        """
        Search for chunks using vector embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            Search results
        """
        search_data = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/search/",
                json=search_data,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the search backend is healthy.
        
        Returns:
            Health status information
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.backend_url}/health",
                    headers=self.headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)} 