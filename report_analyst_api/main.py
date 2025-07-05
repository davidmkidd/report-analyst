"""
Report Analyst FastAPI Application

This module provides REST API endpoints for the report analyst functionality.
Can be deployed independently and integrates with the core report analyst package.
"""

from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging
import os

# Import from core package
try:
    from report_analyst.core.plugins import discover_document_sources, get_available_integrations
    from report_analyst.core.question_loader import get_question_loader
    from report_analyst.core.analyzer import DocumentAnalyzer
except ImportError as e:
    raise ImportError(f"Core report_analyst package not found: {e}")

from .schemas import (
    AnalysisRequest, 
    AnalysisJob, 
    DocumentUpload,
    QuestionSetResponse,
    IntegrationsResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Report Analyst API",
    description="REST API for document analysis using question-based frameworks",
    version="0.1.0"
)

# Global state
_document_sources = None
_question_loader = None
_analyzer = None

def get_document_sources():
    """Get available document sources"""
    global _document_sources
    if _document_sources is None:
        _document_sources = discover_document_sources()
    return _document_sources

def get_question_loader():
    """Get question loader instance"""
    global _question_loader
    if _question_loader is None:
        from report_analyst.core.question_loader import get_question_loader
        _question_loader = get_question_loader()
    return _question_loader

def get_analyzer():
    """Get analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DocumentAnalyzer()
    return _analyzer

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API root endpoint with basic information"""
    integrations = get_available_integrations()
    return {
        "name": "Report Analyst API",
        "version": "0.1.0",
        "description": "REST API for document analysis using question-based frameworks",
        "available_integrations": integrations,
        "endpoints": {
            "upload": "/upload",
            "question_sets": "/question-sets",
            "analyze": "/analyze",
            "health": "/health"
        }
    }

@app.post("/upload", response_model=Dict[str, str])
async def upload_document(upload: DocumentUpload):
    """
    Upload a document for analysis.
    
    Uses the configured document source (local or search backend).
    """
    try:
        sources = get_document_sources()
        
        # Determine which source to use
        source_type = upload.source_type or "local"
        if source_type not in sources:
            raise HTTPException(
                status_code=400, 
                detail=f"Document source '{source_type}' not available. Available: {list(sources.keys())}"
            )
        
        source = sources[source_type]()
        document_id = await source.upload_document(upload.file_path)
        
        return {
            "document_id": document_id,
            "source_type": source_type,
            "status": "uploaded"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/question-sets", response_model=QuestionSetResponse)
async def get_question_sets():
    """Get all available question sets"""
    try:
        question_loader = get_question_loader()
        question_sets = {}
        
        # Get all available question sets
        for set_id in ["tcfd", "s4m", "lucia"]:  # Add more as needed
            question_set = question_loader.get_question_set(set_id)
            if question_set:
                question_sets[set_id] = {
                    "id": set_id,
                    "name": question_set.name,
                    "description": question_set.description,
                    "questions": question_set.questions
                }
        
        return QuestionSetResponse(question_sets=question_sets)
        
    except Exception as e:
        logger.error(f"Failed to get question sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/question-sets/{set_id}", response_model=Dict[str, Any])
async def get_question_set(set_id: str):
    """Get a specific question set"""
    try:
        question_loader = get_question_loader()
        question_set = question_loader.get_question_set(set_id)
        
        if not question_set:
            raise HTTPException(status_code=404, detail=f"Question set '{set_id}' not found")
        
        return {
            "id": set_id,
            "name": question_set.name,
            "description": question_set.description,
            "questions": question_set.questions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get question set {set_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisJob)
async def analyze_document(request: AnalysisRequest):
    """
    Trigger analysis of a document.
    
    This endpoint starts the analysis process and returns a job ID.
    Use the job ID to check status and retrieve results.
    """
    try:
        # For now, this is a simplified implementation
        # In production, this would trigger async processing
        
        analyzer = get_analyzer()
        
        # Create a mock job response
        # TODO: Implement actual async job processing
        job = AnalysisJob(
            job_id=f"job_{request.document_id}_{len(request.selected_questions)}",
            document_id=request.document_id,
            question_set_id=request.question_set_id,
            selected_questions=request.selected_questions,
            status="pending",
            progress=0.0
        )
        
        return job
        
    except Exception as e:
        logger.error(f"Analysis request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/integrations", response_model=IntegrationsResponse)
async def get_integrations():
    """Get information about available integrations"""
    integrations = get_available_integrations()
    sources = list(get_document_sources().keys())
    
    return IntegrationsResponse(
        available_integrations=integrations,
        document_sources=sources
    )

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        # Check core dependencies
        integrations = get_available_integrations()
        sources = get_document_sources()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "core_package": True,
            "question_loader": True,
            "analyzer": True,
            "available_integrations": integrations,
            "document_sources": list(sources.keys())
        }
        
        # Test search backend if available
        if integrations.get("search_backend", False):
            try:
                from report_analyst_search_backend.client import SearchBackendClient
                # You could add a health check call here
                health_status["search_backend_client"] = True
            except Exception as e:
                health_status["search_backend_client"] = False
                health_status["search_backend_error"] = str(e)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        } 