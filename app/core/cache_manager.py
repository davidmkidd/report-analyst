import sqlite3
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import numpy as np
import os

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use the project's storage path
            storage_path = os.getenv('STORAGE_PATH', './storage')
            db_path = str(Path(storage_path) / 'cache' / 'analysis.db')
        
        self.db_path = Path(db_path)
        # Create parent directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing CacheManager with db: {self.db_path}")
        self.init_db()

    def init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    question_id TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,
                    top_k INTEGER,
                    model TEXT,
                    question_set TEXT,
                    result TEXT,
                    created_at TIMESTAMP,
                    UNIQUE(file_path, question_id, chunk_size, chunk_overlap, top_k, model, question_set)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    chunk_text TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    UNIQUE(file_path, chunk_text)
                )
            """)

    def save_analysis(self, 
                     file_path: str,
                     question_id: str,
                     result: Dict,
                     config: Dict):
        """Save analysis result to cache"""
        try:
            # Validate config has all required fields
            required_fields = ['chunk_size', 'chunk_overlap', 'top_k', 'model', 'question_set']
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")
            
            # Validate result can be JSON serialized
            try:
                json_result = json.dumps(result)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Result cannot be JSON serialized: {str(e)}")
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, question_id, chunk_size, chunk_overlap, top_k, model, 
                     question_set, result, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(file_path),
                    question_id,
                    config['chunk_size'],
                    config['chunk_overlap'],
                    config['top_k'],
                    config['model'],
                    config['question_set'],
                    json_result,
                    datetime.now().isoformat()
                ))
                logger.info(f"Saved analysis for {file_path} - {question_id}")
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

    def get_analysis(self, 
                    file_path: str,
                    config: Dict,
                    question_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get analysis results matching the exact configuration"""
        try:
            logger.info(f"Fetching analysis with config: {json.dumps(config, indent=2)}")
            
            # Validate config has all required fields
            required_fields = ['chunk_size', 'chunk_overlap', 'top_k', 'model', 'question_set']
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT question_id, result
                    FROM analysis_cache
                    WHERE file_path = ?
                    AND chunk_size = ?
                    AND chunk_overlap = ?
                    AND top_k = ?
                    AND model = ?
                    AND question_set = ?
                """
                params = [
                    str(file_path),
                    config['chunk_size'],
                    config['chunk_overlap'],
                    config['top_k'],
                    config['model'],
                    config['question_set']
                ]

                if question_ids:
                    placeholders = ','.join('?' * len(question_ids))
                    query += f" AND question_id IN ({placeholders})"
                    params.extend(question_ids)

                logger.info(f"Executing query: {query}")
                logger.info(f"With parameters: {params}")

                results = {}
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                logger.info(f"Found {len(rows)} matching results")
                
                for row in rows:
                    question_id, result_json = row
                    try:
                        results[question_id] = json.loads(result_json)
                        logger.info(f"Loaded result for question {question_id}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode result for {question_id}: {e}")
                        raise

                return results

        except Exception as e:
            logger.error(f"Error retrieving analysis: {str(e)}")
            raise  # Re-raise the exception to make the test pass

    def save_vectors(self, 
                    file_path: str,
                    chunks: List[Dict[str, Any]]):
        """Save vector embeddings for document chunks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for chunk in chunks:
                    conn.execute("""
                        INSERT OR REPLACE INTO vector_cache 
                        (file_path, chunk_text, embedding, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        str(file_path),
                        chunk['text'],
                        chunk['embedding'].tobytes() if 'embedding' in chunk else None,
                        json.dumps(chunk.get('metadata', {})),
                        datetime.now().isoformat()
                    ))
                logger.info(f"Saved {len(chunks)} vectors for {file_path}")
        except Exception as e:
            logger.error(f"Error saving vectors: {str(e)}", exc_info=True)

    def get_vectors(self, file_path: str) -> List[Dict[str, Any]]:
        """Get vector embeddings for a document"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                chunks = []
                for row in conn.execute("""
                    SELECT chunk_text, embedding, metadata
                    FROM vector_cache
                    WHERE file_path = ?
                """, (str(file_path),)):
                    chunks.append({
                        'text': row[0],
                        'embedding': np.frombuffer(row[1]) if row[1] else None,
                        'metadata': json.loads(row[2])
                    })
                logger.info(f"Retrieved {len(chunks)} vectors for {file_path}")
                return chunks
        except Exception as e:
            logger.error(f"Error retrieving vectors: {str(e)}", exc_info=True)
            return []

    def clear_cache(self, file_path: Optional[str] = None):
        """Clear cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if file_path:
                    conn.execute("DELETE FROM analysis_cache WHERE file_path = ?", (str(file_path),))
                    conn.execute("DELETE FROM vector_cache WHERE file_path = ?", (str(file_path),))
                    logger.info(f"Cleared cache for {file_path}")
                else:
                    conn.execute("DELETE FROM analysis_cache")
                    conn.execute("DELETE FROM vector_cache")
                    logger.info("Cleared all cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}", exc_info=True)

    def check_cache_status(self, file_path: str = None):
        """Debug method to check cache contents"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if file_path:
                    logger.info(f"Checking cache for file: {file_path}")
                    cursor = conn.execute("""
                        SELECT DISTINCT chunk_size, chunk_overlap, top_k, model, question_set
                        FROM analysis_cache
                        WHERE file_path = ?
                    """, (str(file_path),))
                else:
                    logger.info("Checking all cache entries")
                    cursor = conn.execute("""
                        SELECT DISTINCT file_path, chunk_size, chunk_overlap, top_k, model, question_set
                        FROM analysis_cache
                    """)
                
                rows = cursor.fetchall()
                logger.info(f"Found {len(rows)} distinct configurations:")
                for row in rows:
                    logger.info(f"Config: {row}")
                
                return rows
                
        except Exception as e:
            logger.error(f"Error checking cache status: {str(e)}", exc_info=True)
            return [] 

    def get_all_answers_by_question_set(self, question_set: str) -> Dict[str, Any]:
        """Get all cached answers for a specific question set"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT question_id, result
                    FROM analysis_cache
                    WHERE question_set = ?
                """, (question_set,))
                
                results = {}
                for row in cursor.fetchall():
                    question_id, result_json = row
                    results[question_id] = json.loads(result_json)
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving answers for question set {question_set}: {e}")
            raise 