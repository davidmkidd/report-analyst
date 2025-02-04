from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Any
import os
from dotenv import load_dotenv
import shutil
import yaml
import logging
import sys
import json
import hashlib
import pickle

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from .prompt_manager import PromptManager
from .storage import LlamaVectorStore

# Setup logging at the top of the file
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not os.getenv("OPENAI_ORGANIZATION"):
    logger.error("OPENAI_ORGANIZATION environment variable is not set")
    raise ValueError("OPENAI_ORGANIZATION environment variable is required")

def log_analysis_step(message: str, level: str = "info"):
    """Helper function to log analysis steps with consistent formatting"""
    log_func = getattr(logger, level)
    log_func(f"[ANALYSIS] {message}")

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def compute_params_hash(params: Dict) -> str:
    """Compute hash of parameters dictionary"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()

class DocumentAnalyzer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.prompt_manager = PromptManager()
        # Use absolute paths for storage, relative to project root
        self.storage_path = Path(__file__).parent.parent.parent / "storage"
        self.cache_path = self.storage_path / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        log_analysis_step(f"Storage path: {self.storage_path.resolve()}", "debug")
        log_analysis_step(f"Cache path: {self.cache_path.resolve()}", "debug")
        
        model_name = os.getenv("OPENAI_API_MODEL", "gpt-4-turbo-preview")
        log_analysis_step(f"Using model: {model_name}")
        
        try:
            self.llm = ChatOpenAI(
                temperature=0,
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORGANIZATION")
            )
            
            # Configure embeddings globally for LlamaIndex
            Settings.embed_model = OpenAIEmbedding(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_API_BASE'),
                model_name=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
                embed_batch_size=100  # Add batch size for embeddings
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20
            )
            
            # Cache parameters
            self.chunk_params = {
                "chunk_size": 500,
                "chunk_overlap": 20
            }
            
            self.embedding_params = {
                "model": "text-embedding-ada-002",  # Default OpenAI embedding model
                "batch_size": 100  # Add batch size to parameters
            }
            
        except Exception as e:
            log_analysis_step(f"Error initializing OpenAI clients: {str(e)}", "error")
            raise
        
        self.questions = self._load_questions()
        self._initialized = True

    def _get_cache_key(self, file_path: str) -> str:
        """Generate a unique cache key for a file based on its path and content hash."""
        try:
            # Use file content hash instead of modification time
            file_hash = compute_file_hash(file_path)
            # Use a shorter hash for parameters
            params_str = f"{self.chunk_params['chunk_size']}_{self.chunk_params['chunk_overlap']}"
            return f"{Path(file_path).stem}_{file_hash[:8]}_{params_str}"
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to generate cache key: {e}")
            # Fallback to a simple key
            return f"{Path(file_path).stem}_fallback"

    def _get_vector_store_collection_name(self, cache_key: str) -> str:
        """Generate a valid collection name from cache key."""
        # Remove any invalid characters and ensure it's not too long
        valid_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in cache_key)
        return valid_name[:63]  # ChromaDB has a limit on collection name length

    def _load_chunks_cache(self, cache_key: str) -> Optional[List]:
        """Load text chunks from cache if available."""
        try:
            cache_file = self.cache_path / f"{cache_key}_chunks.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                # Convert to LlamaIndex Document objects
                chunks = [Document(
                    text=chunk['page_content'],  # LlamaIndex uses text instead of page_content
                    metadata=chunk['metadata']
                ) for chunk in chunk_data]
                logger.info(f"[ANALYSIS] ✓ Cache HIT: Loaded {len(chunks)} chunks from cache")
                return chunks
            logger.info("[ANALYSIS] Cache MISS: No cached chunks found")
            return None
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to load chunks cache: {e}")
            return None

    def _save_chunks_cache(self, cache_key: str, chunks: List) -> None:
        """Save text chunks to cache."""
        try:
            cache_file = self.cache_path / f"{cache_key}_chunks.json"
            # Convert Document objects to serializable format
            serializable_chunks = [{
                'page_content': doc.text,  # Store as page_content for backward compatibility
                'metadata': doc.metadata
            } for doc in chunks]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f)
            logger.info(f"[ANALYSIS] ✓ Cache SAVE: Saved {len(chunks)} chunks to cache")
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to save chunks cache: {e}")

    def _load_vector_store(self, cache_key: str, chunks: List) -> Optional[LlamaVectorStore]:
        """Load vector store from cache if available."""
        try:
            store_dir = self.cache_path / f"{cache_key}_vectors"
            
            if store_dir.exists():
                logger.info(f"[ANALYSIS] Found vector store directory at {store_dir}")
                try:
                    # Load LlamaVectorStore from local files
                    vector_store = LlamaVectorStore(store_dir)
                    # Try to load the store - this will verify if it's valid
                    if vector_store.load():
                        logger.info(f"[ANALYSIS] ✓ Cache HIT: Loaded vector store from cache")
                        return vector_store
                except Exception as inner_e:
                    logger.error(f"[ANALYSIS] Failed to load existing vector store: {inner_e}", exc_info=True)
                    
            logger.info("[ANALYSIS] Cache MISS: No cached vector store found")
            return None
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to load vector store cache: {e}")
            logger.debug(f"Full vector store cache error: {str(e)}", exc_info=True)
            return None

    async def process_document(self, file_path: str, question_ids: List[int] = None) -> AsyncGenerator[Dict, None]:
        """Process document and analyze TCFD questions"""
        log_analysis_step(f"Starting document processing: {file_path}")
        log_analysis_step(f"Processing questions: {question_ids}")
        
        try:
            # Initial status
            yield {"status": "Starting analysis..."}
            
            # Generate cache key
            cache_key = self._get_cache_key(file_path)
            log_analysis_step(f"Using cache key: {cache_key}")
            
            # Get chunks with caching
            log_analysis_step("Checking cache for document chunks...")
            yield {"status": "Loading and chunking document..."}
            
            chunks = self._load_chunks_cache(cache_key)
            if chunks is None:
                # If not in cache, load and process the document
                log_analysis_step("Building new document chunks...")
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                self._save_chunks_cache(cache_key, chunks)
            
            log_analysis_step(f"Using {len(chunks)} text chunks")
            yield {"status": f"✓ Using {len(chunks)} text chunks"}
            
            # Get vector store with caching
            log_analysis_step("Checking cache for vector store...")
            yield {"status": "Creating/loading vector store..."}
            
            store_dir = self.cache_path / f"{cache_key}_vectors"
            
            vectorstore = self._load_vector_store(cache_key, chunks)
            if vectorstore is None:
                log_analysis_step("Building new vector store...")
                yield {"status": "Building vector store (this may take a few minutes)..."}
                
                try:
                    # Ensure clean directory
                    if store_dir.exists():
                        shutil.rmtree(store_dir)
                    store_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create new LlamaVectorStore and add documents
                    vectorstore = LlamaVectorStore(store_dir)
                    vectorstore.add_documents(chunks)
                    
                    # No need to verify by loading again - if add_documents succeeded, it's working
                    log_analysis_step(f"Vector store built and saved successfully")
                    yield {"status": "✓ Vector store built successfully"}
                    
                except Exception as e:
                    error_msg = f"Error building vector store: {str(e)}"
                    log_analysis_step(error_msg, "error")
                    logger.error(f"Full error: {str(e)}", exc_info=True)
                    yield {"error": error_msg}
                    return
            
            log_analysis_step("Vector store ready")
            yield {"status": "✓ Vector store ready"}
            
            # Process each question
            for q_id in question_ids:
                question_key = f"tcfd_{q_id}"
                if question_key not in self.questions:
                    continue
                
                question_data = self.questions[question_key]
                log_analysis_step(f"Processing question {q_id}")
                yield {"status": f"Analyzing question {q_id}"}
                
                try:
                    # Get relevant context using TOP_K=20
                    docs_and_scores = vectorstore.similarity_search(question_data['text'], k=20)
                    docs = [doc for doc, _ in docs_and_scores]
                    scores = [score for _, score in docs_and_scores]
                    
                    context = "\n".join(d.text for d in docs)
                    log_analysis_step(f"Retrieved {len(docs)} relevant chunks for question {q_id}", "debug")
                    
                    # Prepare chunks data for passing to frontend
                    chunks_data = [{"text": d.text, "metadata": d.metadata, "relevance_score": float(s)} 
                                 for d, s in docs_and_scores]
                    
                    # Get LLM response
                    messages = self.prompt_manager.get_analysis_messages(
                        question=question_data['text'],
                        context=context,
                        guidelines=question_data['guidelines']
                    )
                    result = await self.llm.ainvoke(messages)
                    log_analysis_step(f"Got LLM response for question {q_id}", "debug")
                    
                    # Extract JSON from response
                    try:
                        result_text = result.content
                        json_start = result_text.rfind('{')
                        json_end = result_text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            result_text = result_text[json_start:json_end]
                        
                        result_json = json.loads(result_text)
                        
                        # Ensure we have all required keys
                        required_keys = ["ANSWER", "SCORE", "EVIDENCE", "GAPS", "SOURCES"]
                        missing_keys = [key for key in required_keys if key not in result_json]
                        if missing_keys:
                            raise ValueError(f"Missing required keys in response: {missing_keys}")
                        
                        # Return the result in the exact format expected by display code
                        yield {
                            "question_number": q_id,
                            "result": json.dumps({
                                "ANSWER": result_json["ANSWER"],
                                "SCORE": result_json["SCORE"],
                                "EVIDENCE": result_json["EVIDENCE"],
                                "GAPS": result_json["GAPS"],
                                "SOURCES": result_json["SOURCES"],
                                "CHUNKS": chunks_data  # Add chunks to the response
                            })
                        }
                        
                    except Exception as e:
                        log_analysis_step(f"Error processing result for question {q_id}: {str(e)}", "error")
                        yield {
                            "question_number": q_id,
                            "result": json.dumps({
                                "ANSWER": "Error processing analysis response",
                                "SCORE": 0,
                                "EVIDENCE": [],
                                "GAPS": ["Error processing response"],
                                "SOURCES": []
                            })
                        }
                except Exception as e:
                    error_msg = f"Error processing question {q_id}: {str(e)}"
                    log_analysis_step(error_msg, "error")
                    logger.error(f"Full error: {str(e)}", exc_info=True)
                    yield {
                        "question_number": q_id,
                        "result": json.dumps({
                            "ANSWER": f"Error: {error_msg}",
                            "SCORE": 0,
                            "EVIDENCE": [],
                            "GAPS": [error_msg],
                            "SOURCES": []
                        })
                    }
                    
        except Exception as e:
            log_analysis_step(f"Error processing document: {str(e)}", "error")
            yield {"error": f"Failed to process document: {str(e)}"}

    def _load_questions(self) -> dict:
        """Load TCFD questions from YAML files"""
        # Look in app/questionsets first, then try questionsets
        possible_paths = [
            Path(__file__).parent.parent / "questionsets" / "tcfd_questions.yaml",  # app/questionsets
            Path("questionsets") / "tcfd_questions.yaml"  # questionsets in root
        ]
        
        yaml_file = None
        for path in possible_paths:
            if path.exists():
                yaml_file = path
                break
                
        if not yaml_file:
            log_analysis_step(f"Could not find questions file in any of: {[str(p) for p in possible_paths]}", "error")
            return {}
            
        log_analysis_step(f"Loading questions from {yaml_file}", "debug")
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            questions = {}
            
            # Convert the questions list into a structured format
            for q in config.get('questions', []):
                q_id = q.get('id', '')
                if q_id:
                    questions[q_id] = {
                        'text': q.get('text', ''),
                        'guidelines': q.get('guidelines', '')
                    }
            
            log_analysis_step(f"Loaded {len(questions)} questions", "debug")
            return questions 