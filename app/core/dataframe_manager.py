import pandas as pd
import json
from typing import Dict, Tuple, Any, List
import logging

# Setup logging
logger = logging.getLogger(__name__)

def format_list_field(field: Any) -> str:
    """Format list fields for better display"""
    if isinstance(field, str):
        try:
            # Try to parse if it's a string representation of a list
            field = eval(field)
        except:
            return field
            
    if isinstance(field, list):
        formatted_items = []
        for item in field:
            if isinstance(item, dict):
                # Handle evidence items with text and chunk info
                text = item.get('text', '')
                chunk = item.get('chunk', 'Unknown')
                formatted_items.append(f"• {text} [Chunk {chunk}]")
            else:
                formatted_items.append(f"• {str(item)}")
        return "\n".join(formatted_items)
    return str(field)

def extract_evidence_text(evidence: Any) -> str:
    """Extract text from evidence items"""
    if isinstance(evidence, dict):
        text = evidence.get('text', '')
        chunk = evidence.get('chunk', 'Unknown')
        return f"{text} [Chunk {chunk}]"
    return str(evidence)

def create_analysis_dataframes(answers: Dict, report_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create analysis and chunks dataframes from answers"""
    analysis_rows = []
    chunks_rows = []
    
    logger.info(f"Creating dataframes for report: {report_name}")
    logger.info(f"Answers keys: {list(answers.keys())}")
    
    def smart_json_decode(data):
        """Helper function to handle JSON decoding intelligently"""
        if not isinstance(data, str):
            return data
            
        try:
            decoded = json.loads(data)
            # Check if we got a string that might be JSON
            if isinstance(decoded, str):
                try:
                    # Try one more decode
                    return json.loads(decoded)
                except json.JSONDecodeError:
                    # If it fails, return the first decode
                    return decoded
            return decoded
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return data
    
    for question_id, answer_data in answers.items():
        if isinstance(answer_data, dict):
            try:
                # Extract and parse result if needed
                result = smart_json_decode(answer_data.get('result', '{}'))
                config = answer_data.get('config', {})
                
                logger.info(f"\nProcessing question {question_id}")
                logger.info(f"Result keys: {list(result.keys())}")
                
                # Create analysis row with configuration
                analysis_row = {
                    'Report': report_name,
                    'Question ID': answer_data.get('question_id', question_id),
                    'Question': answer_data.get('question_text', ''),
                    'Analysis': result.get('ANSWER', ''),
                    'Score': float(result.get('SCORE', 0)),
                    'Key Evidence': '\n'.join(str(e.get('text', '')) for e in result.get('EVIDENCE', [])),
                    'Gaps': '\n'.join(str(x) for x in result.get('GAPS', [])),
                    'Sources': '\n'.join(str(x) for x in result.get('SOURCES', [])),
                    'Chunk Size': int(config.get('chunk_size', 0)),
                    'Chunk Overlap': int(config.get('chunk_overlap', 0)),
                    'Top K': int(config.get('top_k', 0)),
                    'Model': str(config.get('model', ''))
                }
                analysis_rows.append(analysis_row)
                
                # Create chunk rows with configuration
                chunks = result.get('CHUNKS', [])
                logger.info(f"Found {len(chunks)} chunks")
                for chunk in chunks:
                    chunk_row = {
                        'Report': report_name,
                        'Question ID': answer_data.get('question_id', question_id),
                        'Question': answer_data.get('question_text', ''),
                        'Chunk Text': chunk.get('text', ''),
                        'Vector Similarity': float(chunk.get('relevance_score', 0.0)),
                        'LLM Score': float(chunk.get('computed_score', 0.0)),
                        'Page': str(chunk.get('metadata', {}).get('page_number', '')),
                        'Chunk Size': int(config.get('chunk_size', 0)),
                        'Chunk Overlap': int(config.get('chunk_overlap', 0)),
                        'Top K': int(config.get('top_k', 0)),
                        'Model': str(config.get('model', ''))
                    }
                    chunks_rows.append(chunk_row)
                    
            except Exception as e:
                logger.error(f"Error processing answer {question_id}: {str(e)}")
                logger.error(f"Answer data: {json.dumps(answer_data, indent=2)}")
                continue
    
    analysis_df = pd.DataFrame(analysis_rows)
    chunks_df = pd.DataFrame(chunks_rows)
    
    logger.info(f"Created analysis DataFrame with shape: {analysis_df.shape}")
    logger.info(f"Created chunks DataFrame with shape: {chunks_df.shape}")
    
    return analysis_df, chunks_df

def is_chunk_referenced(position: int, evidence_list: List[Dict]) -> bool:
    """Check if a chunk is referenced in the evidence list"""
    for evidence in evidence_list:
        if evidence.get('chunk') == position:
            return True
    return False

def create_combined_dataframe(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined multi-index dataframe"""
    if analysis_df.empty or chunks_df.empty:
        return pd.DataFrame()
        
    # Create multi-index dataframes using the correct column names
    analysis_indexed = analysis_df.set_index(['Report', 'Question ID'])
    chunks_indexed = chunks_df.set_index(['Report', 'Question ID'])
    
    # Combine the dataframes
    combined_df = pd.concat([analysis_indexed, chunks_indexed], axis=1)
    
    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    return combined_df 