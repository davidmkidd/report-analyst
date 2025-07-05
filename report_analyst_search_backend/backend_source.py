"""
Search Backend Document Source Implementation

This module provides integration with the search backend for document processing
and chunk retrieval, implementing the DocumentSource interface.
"""

import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

try:
    from report_analyst.core.document_sources import DocumentSource, DocumentChunk
except ImportError:
    # Fallback for when core package isn't properly installed
    from abc import ABC, abstractmethod
    
    class DocumentSource(ABC):
        pass
    
    class DocumentChunk:
        def __init__(self, chunk_id: str, chunk_text: str, chunk_metadata: Dict[str, Any], relevance_scores: Optional[Dict[str, float]] = None):
            self.chunk_id = chunk_id
            self.chunk_text = chunk_text
            self.chunk_metadata = chunk_metadata
            self.relevance_scores = relevance_scores or {}

from .client import SearchBackendClient

logger = logging.getLogger(__name__)

class SearchBackendSource(DocumentSource):
    """Document source that uses the search backend for processing"""
    
    def __init__(self, backend_url: str, api_key: Optional[str] = None):
        """
        Initialize search backend integration.
        
        Args:
            backend_url: Base URL of the search backend API
            api_key: Optional API key for authentication
        """
        self.backend_url = backend_url.rstrip('/')
        self.client = SearchBackendClient(backend_url, api_key)
        self._document_mapping = {}  # Map our document IDs to backend resource IDs
    
    async def upload_document(self, file_path: Union[str, Path]) -> str:
        """
        Upload document to search backend.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document ID for subsequent operations
        """
        try:
            # Upload to search backend
            resource_data = await self.client.upload_pdf(file_path)
            resource_id = resource_data["resource_id"]
            
            # Generate our own document ID and map it
            document_id = str(uuid.uuid4())
            self._document_mapping[document_id] = {
                "resource_id": resource_id,
                "filename": Path(file_path).name,
                "status": "uploaded"
            }
            
            logger.info(f"Uploaded document {document_id} -> resource {resource_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise
    
    async def get_chunks(self, document_id: str, configuration: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Get chunks from search backend.
        
        Args:
            document_id: ID of the document
            configuration: Optional chunking configuration
            
        Returns:
            List of document chunks
        """
        if document_id not in self._document_mapping:
            raise ValueError(f"Document {document_id} not found")
        
        resource_id = self._document_mapping[document_id]["resource_id"]
        
        try:
            # Get chunks from search backend
            chunks_data = await self.client.get_chunks(resource_id, configuration)
            
            # Convert to DocumentChunk objects
            chunks = []
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    chunk_id=chunk_data.get("id", f"{document_id}_{len(chunks)}"),
                    chunk_text=chunk_data["chunk_text"],
                    chunk_metadata=chunk_data.get("chunk_metadata", {}),
                    relevance_scores={}
                )
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise
    
    async def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get document processing status from search backend.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Status information
        """
        if document_id not in self._document_mapping:
            return {"status": "not_found"}
        
        resource_id = self._document_mapping[document_id]["resource_id"]
        
        try:
            # Get status from search backend
            resource_status = await self.client.get_resource_status(resource_id)
            
            return {
                "status": resource_status.get("status", "unknown"),
                "document_id": document_id,
                "resource_id": resource_id,
                "backend_status": resource_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for document {document_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from search backend.
        
        Args:
            document_id: ID of the document
            
        Returns:
            True if successful, False otherwise
        """
        if document_id not in self._document_mapping:
            return False
        
        resource_id = self._document_mapping[document_id]["resource_id"]
        
        try:
            # Delete from search backend
            success = await self.client.delete_resource(resource_id)
            
            if success:
                # Remove from our mapping
                del self._document_mapping[document_id]
                logger.info(f"Deleted document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def search_chunks(self, query: str, document_id: Optional[str] = None, top_k: int = 10) -> List[DocumentChunk]:
        """
        Search chunks using the search backend's vector search capabilities.
        
        Args:
            query: Search query
            document_id: Optional document ID to limit search scope
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        try:
            # Use search backend's search functionality
            search_results = await self.client.search_embeddings(query, top_k)
            
            chunks = []
            for result in search_results.get("results", []):
                for chunk_result in result.get("chunks", []):
                    chunk_data = chunk_result["chunk"]
                    similarity = chunk_result["similarity"]
                    
                    # Filter by document if specified
                    if document_id:
                        resource_id = self._document_mapping.get(document_id, {}).get("resource_id")
                        if chunk_data.get("resource_id") != resource_id:
                            continue
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_data["id"],
                        chunk_text=chunk_data["chunk_text"],
                        chunk_metadata=chunk_data.get("chunk_metadata", {}),
                        relevance_scores={"similarity": similarity}
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            raise 