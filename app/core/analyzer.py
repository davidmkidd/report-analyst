from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple
import os
from dotenv import load_dotenv
import shutil
import yaml
import logging
import sys
import json
import hashlib
import pickle
import re
import pandas as pd

from langchain_openai import ChatOpenAI
from llama_index.core import Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionCache

from .prompt_manager import PromptManager
from .storage import LlamaVectorStore
from .cache_manager import CacheManager
import numpy as np

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
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.prompt_manager = PromptManager()
        # Use absolute paths for storage
        self.storage_path = Path(__file__).parent.parent.parent / "storage"
        self.cache_path = self.storage_path / "cache"
        self.llm_cache_path = self.storage_path / "llm_cache"
        
        # Create cache directories
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.llm_cache_path.mkdir(parents=True, exist_ok=True)
        
        log_analysis_step(f"Storage path: {self.storage_path.resolve()}", "debug")
        log_analysis_step(f"Cache path: {self.cache_path.resolve()}", "debug")
        log_analysis_step(f"LLM cache path: {self.llm_cache_path.resolve()}", "debug")
        
        # Set default question set
        self.question_set = "tcfd"
        self.questions = self._load_questions()
        
        # Use model from environment variables as default
        self.default_model = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo-1106")
        log_analysis_step(f"Using default model from env: {self.default_model}")
        
        try:
            # Initialize LLM with caching
            self.llm = OpenAI(
                model=self.default_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_API_BASE"),
                cache_dir=str(self.llm_cache_path),
            )
            
            # Configure embeddings globally for LlamaIndex
            Settings.embed_model = OpenAIEmbedding(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_API_BASE'),
                model_name=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
                embed_batch_size=100
            )
            
            # Initialize caching
            self.use_cache = True  # Default to True, can be overridden
            Settings.ingestion_cache = IngestionCache(
                cache_dir=str(self.llm_cache_path),
                cache_type="local"
            )
            
            self.text_splitter = SentenceSplitter(
                chunk_size=500,
                chunk_overlap=20
            )
            
            # Cache parameters
            self.chunk_params = {
                "chunk_size": 500,
                "chunk_overlap": 20,
                "top_k": 5
            }
            
            self.embedding_params = {
                "model": "text-embedding-ada-002",
                "batch_size": 100
            }
            
        except Exception as e:
            log_analysis_step(f"Error initializing OpenAI clients: {str(e)}", "error")
            raise
        
        # Add a cache for loaded answers
        self._answers_cache = {}
        
        self.cache_manager = CacheManager()
        logger.info("Initialized DocumentAnalyzer with cache manager")
        
        self._initialized = True

    def _get_cache_key(self, file_path: str) -> str:
        """Generate a unique cache key based on file and all analysis parameters."""
        try:
            params_str = (f"cs{self.chunk_params['chunk_size']}_"
                         f"ov{self.chunk_params['chunk_overlap']}_"
                         f"tk{self.chunk_params['top_k']}_"
                         f"m{self.llm.model}_"  # Include LLM model
                         f"qs{self.question_set}")  # Include question set
            return f"{Path(file_path).stem}_{params_str}"
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to generate cache key: {e}")
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

    async def score_chunk_relevance(self, question: str, chunk_text: str) -> float:
        """Score the relevance of a chunk to a question using LLM."""
        if not self.use_cache:
            Settings.ingestion_cache = None
            
        log_analysis_step(f"Computing relevance score for chunk: {chunk_text[:100]}...")
        
        try:
            response = await self.llm.acomplete(prompt=f"""As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are tasked with evaluating text fragments for their usefulness in answering specific TCFD questions.

Your task is to score the relevance and quality of evidence in each text fragment. Consider:

1. Specificity and Concreteness:
   - Quantitative data and specific metrics (highest value)
   - Concrete policies and procedures
   - Specific commitments with timelines
   - General statements or vague claims (lowest value)

2. Evidence Quality:
   - Verifiable data and third-party verification
   - Clear methodologies and frameworks
   - Specific examples and case studies
   - Unsubstantiated claims (lowest value)

3. Direct Relevance:
   - Direct answers to the question components
   - Related but indirect information
   - Contextual background
   - Unrelated information (lowest value)

4. Disclosure Quality:
   - Comprehensive and transparent disclosure
   - Balanced reporting (both positive and negative)
   - Clear acknowledgment of limitations
   - Potential greenwashing or selective disclosure (lowest value)

Score from 0.0 to 1.0 where:
0.0 = Not useful (generic statements, unrelated content)
0.3 = Contains relevant context but no specific evidence
0.6 = Contains useful specific information but requires additional context
1.0 = Contains critical evidence or specific details that directly answer the question

Question: {question}

Text to evaluate:
{chunk_text}

Output only the numeric score (0.0-1.0):""")
            
            score = float(response.text.strip())
            score = max(0.0, min(1.0, score))
            log_analysis_step(f"Computed relevance score: {score:.4f}")
            return score
            
        except Exception as e:
            log_analysis_step(f"Error scoring chunk relevance: {str(e)}", "error")
            return 0.0

    async def score_chunk_relevance_batch(self, question: str, chunks: List[Dict], single_call: bool = True) -> List[float]:
        """Score a batch of chunks using LLM.
        
        Args:
            question: The question being analyzed
            chunks: List of chunks to score
            single_call: If True, score all chunks in one API call. If False, score each chunk individually.
        """
        try:
            if single_call:
                # Batch scoring - all chunks in one call
                chunks_text = "\n\n".join([
                    f"[CHUNK {i+1}]\n{chunk['text']}"
                    for i, chunk in enumerate(chunks)
                ])
                
                response = await self.llm.acomplete(prompt=f"""As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are tasked with evaluating text fragments for their usefulness in answering specific TCFD questions.

Your task is to score the relevance and quality of evidence in each text fragment marked as [CHUNK X]. Consider:

1. Specificity and Concreteness:
   - Quantitative data and specific metrics (highest value)
   - Concrete policies and procedures
   - Specific commitments with timelines
   - General statements or vague claims (lowest value)

2. Evidence Quality:
   - Verifiable data and third-party verification
   - Clear methodologies and frameworks
   - Specific examples and case studies
   - Unsubstantiated claims (lowest value)

3. Direct Relevance:
   - Direct answers to the question components
   - Related but indirect information
   - Contextual background
   - Unrelated information (lowest value)

4. Disclosure Quality:
   - Comprehensive and transparent disclosure
   - Balanced reporting (both positive and negative)
   - Clear acknowledgment of limitations
   - Potential greenwashing or selective disclosure (lowest value)

For each chunk marked [CHUNK X], provide a score from 0.0 to 1.0 where:
0.0 = Not useful (generic statements, unrelated content)
0.3 = Contains relevant context but no specific evidence
0.6 = Contains useful specific information but requires additional context
1.0 = Contains critical evidence or specific details that directly answer the question

Question: {question}

Text fragments to evaluate:
{chunks_text}

Output only the scores, one per line, in order:""")
                
                # Parse scores from response
                try:
                    scores = [float(score.strip()) for score in response.text.strip().split('\n')]
                    if len(scores) != len(chunks):
                        raise ValueError(f"Got {len(scores)} scores for {len(chunks)} chunks")
                    return scores
                except Exception as e:
                    log_analysis_step(f"Error parsing batch scores: {str(e)}", "error")
                    return [0.0] * len(chunks)
                
            else:
                # Individual scoring - one API call per chunk
                scores = []
                for i, chunk in enumerate(chunks):
                    score = await self.score_chunk_relevance(question, chunk['text'])
                    scores.append(score)
                    log_analysis_step(f"Scored chunk {i+1}/{len(chunks)}: {score:.2f}")
                return scores
                
        except Exception as e:
            log_analysis_step(f"Error in batch scoring: {str(e)}", "error")
            return [0.0] * len(chunks)

    def _load_cached_answers(self, file_path: str) -> Dict:
        """Load cached answers for a file with exact configuration match"""
        try:
            # Log current configuration
            logger.info(f"Current configuration:")
            logger.info(f"- Chunk size: {self.chunk_params['chunk_size']}")
            logger.info(f"- Overlap: {self.chunk_params['chunk_overlap']}")
            logger.info(f"- Top K: {self.chunk_params['top_k']}")
            logger.info(f"- Model: {self.llm.model}")
            logger.info(f"- Question set: {self.question_set}")
            
            # Log cache directory and available files
            logger.info(f"Cache directory: {self.cache_path}")
            cache_files = list(self.cache_path.glob("*.json"))
            logger.info(f"Available cache files ({len(cache_files)}):")
            for cf in cache_files:
                logger.info(f"- {cf.name}")
            
            # Generate cache key for current configuration
            cache_key = f"cs{self.chunk_params['chunk_size']}_ov{self.chunk_params['chunk_overlap']}_tk{self.chunk_params['top_k']}_m{self.llm.model}_qs{self.question_set}"
            file_stem = Path(file_path).stem
            cache_file = Path(self.cache_path) / f"{file_stem}_{cache_key}.json"
            
            logger.info(f"Looking for cache file: {cache_file}")
            
            if not cache_file.exists():
                logger.info(f"No cache file found for current configuration")
                return {}
                
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.info(f"Loaded cache data with keys: {list(cached_data.keys())}")
                logger.info(f"Cache data structure: {json.dumps(cached_data, indent=2)[:500]}...")  # Show first 500 chars
                return cached_data
                
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return {}

    def _validate_cache_filename(self, filename: str) -> bool:
        """Check if cache filename follows the required pattern."""
        # Pattern: filename_cs{num}_ov{num}_tk{num}_m{model}_qs{set}.json
        pattern = r'^.+_cs\d+_ov\d+_tk\d+_m[^_]+_qs[^_]+\.json$'
        return bool(re.match(pattern, filename))

    def _save_cached_answers(self, file_path: str, answers: Dict) -> None:
        """Save answers using the parameter-based format and update memory cache"""
        try:
            cache_key = self._get_cache_key(file_path)
            cache_path = self.cache_path / f"{cache_key}.json"
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(answers, f, indent=2)
            
            # Update memory cache
            self._answers_cache[cache_key] = answers
            logger.info(f"[ANALYSIS] ✓ Cache SAVE: Saved answers to {cache_path}")
            
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to save answers: {e}")

    async def process_document(self, file_path: str, question_ids: List[str]) -> AsyncGenerator[Dict, None]:
        """Process document with caching"""
        try:
            # Get current configuration
            config = {
                'chunk_size': self.chunk_params['chunk_size'],
                'chunk_overlap': self.chunk_params['chunk_overlap'],
                'top_k': self.chunk_params['top_k'],
                'model': self.llm.model,
                'question_set': self.question_set
            }
            logger.info(f"Processing document with config: {config}")

            # Check cache first
            cached_results = self.cache_manager.get_analysis(
                file_path=file_path,
                config=config,
                question_ids=question_ids
            )

            # Return cached results immediately
            for qid, result in cached_results.items():
                yield {
                    'status': 'cached',
                    'question_id': qid,
                    'result': result
                }

            # Process remaining questions
            remaining_questions = [q for q in question_ids if q not in cached_results]
            if not remaining_questions:
                logger.info("All questions found in cache")
                return

            # Get or compute document chunks
            chunks = self.cache_manager.get_vectors(file_path)
            if not chunks:
                logger.info("No cached vectors found, processing document")
                yield {'status': 'processing', 'message': 'Creating document chunks...'}
                chunks = self._create_chunks(file_path)
                self.cache_manager.save_vectors(file_path, chunks)

            # Process each uncached question
            for question_id in remaining_questions:
                try:
                    logger.info(f"Processing question {question_id}")
                    yield {'status': 'processing', 'message': f'Analyzing question {question_id}...'}

                    result = await self._analyze_question(question_id, chunks)
                    
                    # Save to cache
                    self.cache_manager.save_analysis(
                        file_path=file_path,
                        question_id=question_id,
                        result=result,
                        config=config
                    )

                    yield {
                        'status': 'complete',
                        'question_id': question_id,
                        'result': result
                    }

                except Exception as e:
                    logger.error(f"Error processing question {question_id}: {str(e)}", exc_info=True)
                    yield {
                        'status': 'error',
                        'question_id': question_id,
                        'error': str(e)
                    }

        except Exception as e:
            logger.error(f"Error in process_document: {str(e)}", exc_info=True)
            yield {'status': 'error', 'error': str(e)}

    def _create_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Create document chunks with embeddings"""
        try:
            logger.info(f"Creating chunks for {file_path}")
            reader = PyMuPDFReader()
            docs = reader.load(file_path=file_path)
            
            # Convert the documents to text and create new Document objects
            text_chunks = []
            for doc in docs:
                nodes = self.text_splitter.split_text(doc.text)
                text_chunks.extend([
                    Document(text=chunk, metadata=doc.metadata)
                    for chunk in nodes
                ])
            
            logger.info(f"Created {len(text_chunks)} chunks")
            
            # Convert to the expected dictionary format with embeddings
            chunks_data = []
            for chunk in text_chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "similarity": 0.0,  # Will be populated during analysis
                    "computed_score": 0.0  # Will be populated during analysis
                }
                chunks_data.append(chunk_dict)
            
            # Save chunks to cache
            self.cache_manager.save_vectors(file_path, chunks_data)
            
            return chunks_data
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}", exc_info=True)
            raise

    async def _analyze_question(self, question_id: str, chunks: List[Dict]) -> Dict:
        """Analyze a single question"""
        try:
            # Get question data
            question_data = self.questions.get(question_id)
            if not question_data:
                raise ValueError(f"Question {question_id} not found")
            
            # Sort chunks by similarity score
            sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity', 0.0), reverse=True)
            top_chunks = sorted_chunks[:self.chunk_params['top_k']]
            context = "\n".join(chunk['text'] for chunk in top_chunks)
            
            # Get LLM response with sorted chunks data
            messages = self.prompt_manager.get_analysis_messages(
                question=question_data['text'],
                context=context,
                guidelines=question_data['guidelines'],
                chunks_data=top_chunks
            )
            
            # Convert messages to a single prompt for LlamaIndex OpenAI
            prompt = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages
            ])
            
            result = await self.llm.acomplete(prompt=prompt)
            
            # Extract JSON from response
            try:
                result_text = result.text.strip()
                
                # Find the first { and last } to extract just the JSON object
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
                    result_text = result_text.replace(',}', '}')
                    result_text = result_text.replace('```json', '').replace('```', '')
                    
                    result_json = json.loads(result_text)
                    
                    # Ensure we have all required keys
                    required_keys = ["ANSWER", "SCORE", "EVIDENCE", "GAPS", "SOURCES"]
                    missing_keys = [key for key in required_keys if key not in result_json]
                    if missing_keys:
                        raise ValueError(f"Missing required keys in response: {missing_keys}")
                    
                    # Create final result dictionary
                    result_dict = {
                        "answer": result_json["ANSWER"],
                        "score": result_json["SCORE"],
                        "evidence": result_json["EVIDENCE"],
                        "gaps": result_json["GAPS"],
                        "sources": result_json["SOURCES"],
                        "chunks": [
                            {
                                "text": chunk["text"],
                                "similarity": float(chunk.get("similarity", 0.0)),
                                "llm_score": float(chunk.get("computed_score", 0.0))
                            }
                            for chunk in top_chunks
                        ]
                    }
                    
                    return result_dict
                else:
                    raise ValueError("No valid JSON object found in response")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}\nResponse text: {result_text[:200]}")
                raise
                
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}", exc_info=True)
            raise

    def _load_questions(self) -> dict:
        """Load questions from YAML files"""
        # Look for question set file in multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "questionsets" / f"{self.question_set}_questions.yaml",  # app/questionsets
            Path(__file__).parent.parent.parent / "questionsets" / f"{self.question_set}_questions.yaml",  # project root
            Path.cwd() / "questionsets" / f"{self.question_set}_questions.yaml"  # current working directory
        ]
        
        log_analysis_step(f"Looking for {self.question_set}_questions.yaml in:")
        for path in possible_paths:
            log_analysis_step(f"- {path.resolve()}")
        
        yaml_file = None
        for path in possible_paths:
            if path.exists():
                yaml_file = path
                log_analysis_step(f"✓ Found questions file at: {path.resolve()}")
                break
                
        if not yaml_file:
            log_analysis_step(f"Could not find questions file for {self.question_set} in any of: {[str(p) for p in possible_paths]}", "error")
            return {}
            
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                log_analysis_step(f"Loaded YAML content: {str(config)[:200]}...")  # Show first 200 chars
                
                questions = {}
                # Convert the questions list into a structured format
                for q in config.get('questions', []):
                    q_id = q.get('id', '')
                    if q_id:
                        questions[q_id] = {
                            'text': q.get('text', ''),
                            'guidelines': q.get('guidelines', '')
                        }
                        log_analysis_step(f"Added question {q_id}: {questions[q_id]['text'][:50]}...")
                
                log_analysis_step(f"✓ Loaded {len(questions)} questions for {self.question_set}")
                log_analysis_step(f"Available question IDs: {list(questions.keys())}")
                return questions
        except Exception as e:
            log_analysis_step(f"Error loading questions: {str(e)}", "error")
            logger.exception("Full error:")  # This will log the full traceback
            return {}

    def get_question_by_number(self, number: int) -> Optional[Dict]:
        """Get question data by its number."""
        try:
            question_key = f"{self.question_set}_{number}"
            return self.questions.get(question_key)
        except Exception as e:
            log_analysis_step(f"Error getting question {number}: {str(e)}", "error")
            logger.exception("Full error:")
            return None

    def update_parameters(self, chunk_size: int, chunk_overlap: int, top_k: int):
        """Update analysis parameters and recreate text splitter."""
        logger.info(f"Updating parameters: size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}")
        
        self.chunk_params = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': top_k
        }
        
        # Recreate text splitter with new parameters
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Updated parameters and recreated text splitter")

    def update_llm_model(self, model_name: str):
        """Update the LLM model."""
        log_analysis_step(f"Updating LLM model to: {model_name}")
        
        # Initialize LLM with caching
        self.llm = OpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            cache_dir=str(self.llm_cache_path),
        )

    def get_all_cached_answers(self, question_set: str) -> Dict[str, Any]:
        """Get all cached answers for a question set"""
        return self.cache_manager.get_all_answers_by_question_set(question_set)

    def update_question_set(self, question_set: str):
        """Update the question set and reload questions."""
        self.question_set = question_set
        self.questions = self._load_questions() 

    def _parse_config_from_filename(self, filename: str) -> Dict[str, Any]:
        """Parse configuration parameters from a cache filename.
        
        Args:
            filename: The filename (without extension) to parse
            
        Returns:
            Dict containing the parsed configuration parameters
        """
        config = {
            'chunk_size': 500,  # Default values
            'overlap': 20,
            'top_k': 5,
            'model': 'gpt-3.5-turbo-1106',
            'question_set': 'tcfd'
        }
        
        try:
            # Split filename into parts
            parts = filename.split('_')
            
            for part in parts:
                if part.startswith('cs'):
                    config['chunk_size'] = int(part[2:])
                elif part.startswith('ov'):
                    config['overlap'] = int(part[2:])
                elif part.startswith('tk'):
                    config['top_k'] = int(part[2:])
                elif part.startswith('m'):
                    config['model'] = part[1:]
                elif part.startswith('qs'):
                    config['question_set'] = part[2:]
                    
            return config
            
        except Exception as e:
            logger.warning(f"Error parsing config from filename {filename}: {e}")
            return config 

def create_analysis_dataframes(results: Dict) -> pd.DataFrame:
    """Create analysis dataframes with proper type handling"""
    analysis_rows = []
    
    # Get analyzer instance to access questions
    analyzer = DocumentAnalyzer()
    questions = analyzer.questions
    
    for question_id, data in results.items():
        # Skip empty results
        if not data:
            continue
            
        # Get question text from analyzer's questions data
        question_text = questions.get(question_id, {}).get('text', f'Question {question_id}')
        
        # Convert lists to strings and ensure proper types
        row = {
            'Question ID': str(question_id),
            'Question': str(question_text),
            'Analysis': str(data.get('ANSWER', '')),
            'Score': float(data.get('SCORE', 0)),
            'Key Evidence': ', '.join(str(x) for x in data.get('EVIDENCE', [])),
            'Gaps': ', '.join(str(x) for x in data.get('GAPS', [])),
            'Sources': ', '.join(str(x) for x in data.get('SOURCES', []))
        }
        analysis_rows.append(row)
    
    # Create DataFrame with explicit dtypes
    df = pd.DataFrame(analysis_rows)
    if not df.empty:
        df = df.astype({
            'Question ID': 'string',
            'Question': 'string',
            'Analysis': 'string',
            'Score': 'float64',
            'Key Evidence': 'string',
            'Gaps': 'string',
            'Sources': 'string'
        })
    
    return df 