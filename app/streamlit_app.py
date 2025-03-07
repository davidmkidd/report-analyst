import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import yaml
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from pathlib import Path
import sys
from dotenv import load_dotenv
import traceback
import time
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import glob  # Add at the top of the file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from other libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

def log_analysis_step(message: str, level: str = "info"):
    """Helper function to log analysis steps with consistent formatting"""
    log_func = getattr(logger, level)
    log_func(f"[ANALYSIS] {message}")

# Add the report-analyst directory to the Python path
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
logger.info(f"Added {current_dir} to Python path")

# Keep relative imports
from app.core.analyzer import DocumentAnalyzer
from app.core.prompt_manager import PromptManager
from app.core.dataframe_manager import create_analysis_dataframes, create_combined_dataframe

# Load environment variables
load_dotenv()
logger.info("Loaded environment variables")

LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

question_sets = {
    "tcfd": {
        "name": "TCFD Questions",
        "description": "Task Force on Climate-related Financial Disclosures questions"
    },
    "s4m": {
        "name": "S4M Questions",
        "description": "Sustainability for Management questions"
    }
}

class ReportAnalyzer:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize the real document analyzer
        self.analyzer = DocumentAnalyzer()
        self.prompt_manager = PromptManager()
        
    def load_question_set(self, question_set: str) -> Dict:
        """Load questions from the specified question set file"""
        question_file = Path(__file__).parent / "questionsets" / f"{question_set}_questions.yaml"
        try:
            with open(question_file, 'r') as f:
                data = yaml.safe_load(f)
                # Create questions with proper IDs
                questions = {}
                for i, q in enumerate(data['questions'], 1):  # Start from 1
                    q_id = f"{question_set}_{i}"
                    questions[q_id] = q
                    # Add the ID to the question data for reference
                    q['id'] = q_id
                    # Add numeric ID for easier mapping
                    q['number'] = i
                return {
                    "questions": questions,
                    "name": data.get('name', f"{question_set.upper()} Questions"),
                    "description": data.get('description', '')
                }
        except Exception as e:
            logger.error(f"Failed to load questions from {question_file}: {str(e)}")
            return {
                "questions": {},
                "name": "",
                "description": ""
            }
    
    async def analyze_document(self, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True, force_recompute: bool = False) -> AsyncGenerator[Dict, None]:
        """Analyze a document using the provided questions"""
        try:
            log_analysis_step(f"Starting analysis of document: {file_path}")
            log_analysis_step(f"Selected questions: {selected_questions}")
            log_analysis_step(f"LLM scoring enabled: {use_llm_scoring}")
            
            # Update analyzer with the current questions
            self.analyzer.questions = questions
            
            # Convert selected question IDs to numbers for the analyzer
            selected_numbers = [questions[q_id]['number'] for q_id in selected_questions]
            
            # Get the question set prefix from the first selected question
            question_set = selected_questions[0].split('_')[0] if selected_questions else "tcfd"
            self.analyzer.question_set = question_set
            
            # Pass use_llm_scoring to process_document
            async for result in self.analyzer.process_document(
                file_path, 
                selected_numbers, 
                use_llm_scoring, 
                single_call,
                force_recompute
            ):
                # Convert question number back to ID if needed
                if 'question_number' in result:
                    result['question_id'] = f"{question_set}_{result['question_number']}"
                yield result
            
        except Exception as e:
            log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
            st.error(f"Error analyzing document: {str(e)}")
            yield {"error": f"Error analyzing document: {str(e)}"}

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temp directory"""
    try:
        if uploaded_file is None:
            logger.warning("No file was uploaded")
            return None
            
        # If it's already a path, just return it
        if isinstance(uploaded_file, (str, Path)):
            return str(uploaded_file)
            
        # Check if file was already saved in this session
        file_key = f"saved_file_{uploaded_file.name}"
        if file_key in st.session_state:
            return st.session_state[file_key]
            
        # Otherwise, handle it as an UploadedFile
        file_path = Path("temp") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Successfully saved file: {file_path}")
        
        # Store the path in session state
        st.session_state[file_key] = str(file_path)
        # Reset file processing flag when a new file is saved
        st.session_state.file_processed = False
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        st.error(f"Error saving file: {str(e)}")
        return None

def display_dataframes(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame):
    """Display only the dataframes without download buttons"""
    # Main Analysis Table (only once)
    st.subheader("Analysis Results")
    st.dataframe(
        analysis_df,
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score",
                help="Analysis score out of 10",
                min_value=0,
                max_value=10,
                format="%.1f"
            ),
            "Analysis": st.column_config.TextColumn(
                "Analysis",
                width="large"
            ),
            "Key Evidence": st.column_config.TextColumn(
                "Key Evidence",
                width="medium"
            )
        }
    )
    
    # Document Chunks Table (only once)
    st.subheader("Document Chunks")
    st.dataframe(
        chunks_df,
        use_container_width=True,
        column_config={
            "Vector Similarity": st.column_config.NumberColumn(
                "Vector Similarity",
                format="%.3f"
            ),
            "LLM Score": st.column_config.NumberColumn(
                "LLM Score",
                format="%.3f"
            ),
            "Chunk Text": st.column_config.TextColumn(
                "Chunk Text",
                width="large"
            )
        }
    )

def convert_df(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')

def display_download_buttons(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame, file_key: str):
    """Display download buttons for analysis results"""
    # Generate unique timestamp for this render
    timestamp = int(time.time() * 1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = convert_df(analysis_df)
        st.download_button(
            label="Download Analysis Results",
            data=csv,
            file_name=f"analysis_{file_key}.csv",
            mime="text/csv",
            key=f"download_analysis_{file_key}_{timestamp}"
        )
    
    with col2:
        csv = convert_df(chunks_df)
        st.download_button(
            label="Download Chunks Data",
            data=csv,
            file_name=f"chunks_{file_key}.csv",
            mime="text/csv",
            key=f"download_chunks_{file_key}_{timestamp}"
        )

def generate_file_key(file_path: str, st) -> str:
    """Generate a cache file key from file path and settings"""
    return (f"{Path(file_path).name}_"
            f"cs{st.session_state.new_chunk_size}_"
            f"ov{st.session_state.new_overlap}_"
            f"tk{st.session_state.new_top_k}_"
            f"m{st.session_state.new_llm_model}")

async def analyze_document_and_display(analyzer, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True, force_recompute: bool = False):
    """Analyze document and display results as they come in"""
    try:
        selected_questions_list = list(selected_questions) if selected_questions else []
        question_set = selected_questions_list[0].split('_')[0] if selected_questions_list else "tcfd"
        
        # Use the helper function to generate file key
        file_key = generate_file_key(file_path, st)
        
        # Initialize or clear results if question set changed
        if ('results' not in st.session_state or 
            'current_question_set' not in st.session_state or 
            st.session_state.current_question_set != question_set):
            st.session_state.results = {"answers": {}}
            st.session_state.current_question_set = question_set
            st.session_state.analyzed_files = set()
            
        # Create status placeholder
        status_placeholder = st.empty()
            
        log_analysis_step(f"Starting analysis with question set: {question_set}")
        
        # Load cached answers first
        cached_answers = {} if force_recompute else analyzer.analyzer._load_cached_answers(file_path)
        
        if cached_answers:
            log_analysis_step(f"Found {len(cached_answers)} cached answers for {file_key}")
            # Show cache info
            st.info(f"📁 Loading results from cache: {file_key}")
            
            # Filter cached answers to only include current question set
            cached_answers = {k: v for k, v in cached_answers.items() 
                            if k.startswith(question_set)}
            
            if cached_answers:
                # Immediately update results with cached answers
                for q_id, answer in cached_answers.items():
                    st.session_state.results["answers"][q_id] = answer
                
                # Update display with cached results
                analysis_df, chunks_df = create_analysis_dataframes(
                    st.session_state.results["answers"], 
                    file_key
                )
                st.session_state.analysis_df = analysis_df
                st.session_state.chunks_df = chunks_df
        
        # Determine which questions need processing
        questions_to_process = [q_id for q_id in selected_questions_list 
                              if force_recompute or q_id not in cached_answers]
        
        if questions_to_process:
            log_analysis_step(f"Processing {len(questions_to_process)} uncached questions...")
            
            # Update analyzer with question set
            analyzer.analyzer.question_set = question_set
            
            # Process only uncached questions
            async for result in analyzer.analyze_document(
                file_path, 
                questions,
                questions_to_process,
                use_llm_scoring, 
                single_call,
                force_recompute
            ):
                if "error" in result:
                    log_analysis_step(f"Error received from analyzer: {result['error']}", "error")
                    st.error(f"Analysis error: {result['error']}")
                    continue
                
                if "status" in result:
                    status_placeholder.write(result["status"])
                    continue
                    
                question_id = result.get('question_id')
                if question_id is None:
                    continue
                
                # Store results
                st.session_state.results["answers"][question_id] = result
                
                # Update display
                analysis_df, chunks_df = create_analysis_dataframes(
                    st.session_state.results["answers"], 
                    file_key
                )
                
                st.session_state.analysis_df = analysis_df
                st.session_state.chunks_df = chunks_df
        else:
            log_analysis_step("All selected questions have cached answers")
            # Show success message for cached results
            st.success(f"✓ All {len(selected_questions_list)} selected questions loaded from cache")
        
        # Mark this file as analyzed with current configuration
        if 'analyzed_files' not in st.session_state:
            st.session_state.analyzed_files = set()
        st.session_state.analyzed_files.add(file_key)
        
        # Clear status and mark as complete
        status_placeholder.empty()
        st.session_state.analysis_complete = True
        
    except Exception as e:
        log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
        log_analysis_step(traceback.format_exc(), "error")
        st.error(f"Error during analysis: {str(e)}")

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def display_final_results(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame):
    """Display the final results including both tables"""
    # Analysis Results
    st.subheader("Analysis Results")
    st.dataframe(
        analysis_df,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score",
                help="Analysis score out of 10",
                min_value=0,
                max_value=10,
                format="%.1f"
            ),
            "Analysis": st.column_config.TextColumn(
                "Analysis",
                width="large"
            ),
            "Key Evidence": st.column_config.TextColumn(
                "Key Evidence",
                width="medium"
            )
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Document Chunks
    st.subheader("Document Chunks")
    st.dataframe(
        filter_dataframe(chunks_df),  # Apply the filter function
        column_config={
            "Question ID": st.column_config.SelectboxColumn(
                "Question ID",
                help="The question this chunk belongs to",
                width="medium",
                options=chunks_df["Question ID"].unique().tolist()
            ),
            "Vector Similarity": st.column_config.NumberColumn(
                "Vector Similarity",
                help="Similarity score between chunk and question",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "LLM Score": st.column_config.NumberColumn(
                "LLM Score",
                help="LLM-computed relevance score",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "Chunk Text": st.column_config.TextColumn(
                "Chunk Text",
                help="Text content of the chunk",
                width="large"
            ),
            "Evidence Reference": st.column_config.CheckboxColumn(
                "Used as Evidence",
                help="Whether this chunk was referenced in the analysis"
            ),
            "Position in Question": st.column_config.NumberColumn(
                "Position",
                help="Position of chunk within question results",
                min_value=0
            )
        },
        use_container_width=True,
        hide_index=False
    )

def load_question_sets() -> Dict[str, str]:
    """Load all available question sets and their descriptions"""
    question_sets = {}
    question_sets_dir = Path(__file__).parent / "questionsets"
    
    for yaml_file in question_sets_dir.glob("*_questions.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                # Get set name from filename (e.g., 'tcfd' from 'tcfd_questions.yaml')
                set_id = yaml_file.stem.replace('_questions', '')
                question_sets[set_id] = {
                    'name': data.get('name', set_id.upper()),
                    'description': data.get('description', '')
                }
        except Exception as e:
            logger.error(f"Error loading question set {yaml_file}: {e}")
    
    return question_sets

def get_uploaded_files_history() -> List[Dict]:
    """Get list of previously uploaded files from temp directory"""
    temp_dir = Path("temp")
    if not temp_dir.exists():
        return []
    
    files = []
    for file in temp_dir.glob("*.pdf"):
        # Verify file exists and is not empty
        if file.exists() and file.stat().st_size > 0:
            files.append({
                'name': file.name,
                'path': str(file.resolve()),  # Get absolute path
                'date': file.stat().st_mtime,
                'size': file.stat().st_size
            })
            logger.info(f"Found file: {file.name}, size: {file.stat().st_size} bytes")
    
    # Sort by most recent first
    return sorted(files, key=lambda x: x['date'], reverse=True)

def get_all_cached_answers(question_set: str) -> Dict:
    """Get all cached answers for a question set from all reports"""
    analyzer = DocumentAnalyzer()  # Get the singleton instance
    logger.info(f"[ANALYSIS] Looking for cached answers in: {analyzer.cache_path}")
    
    # Log all files in cache directory first
    all_files = glob.glob(f"{analyzer.cache_path}/*")
    logger.info(f"All files in cache directory: {all_files}")
    
    # Get all cache files for this question set using the correct pattern
    pattern = f"{analyzer.cache_path}/*_qs{question_set}.json"
    logger.info(f"Searching with pattern: {pattern}")
    cache_files = glob.glob(pattern)
    logger.info(f"Found cache files: {cache_files}")
    
    all_answers = {}
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                # Extract file key from cache filename - just get the document name part
                file_name = Path(cache_file).stem
                # Get everything before the first '_cs' which marks the start of parameters
                file_key = file_name.split('_cs')[0]
                answers = json.load(f)
                all_answers[file_key] = answers
                logger.info(f"Successfully loaded cached answers for {file_key} from {cache_file}")
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            continue
    
    logger.info(f"Total cached answers loaded: {len(all_answers)}")
    return all_answers

def create_coverage_matrix(question_set: str) -> pd.DataFrame:
    """Create a matrix showing which reports have answers for which questions"""
    all_answers = get_all_cached_answers(question_set)
    
    # Get all unique question IDs
    all_questions = set()
    for answers in all_answers.values():
        all_questions.update(answers.keys())
    
    # Create matrix
    matrix_data = []
    for report_name, answers in all_answers.items():
        row = {'Report': report_name}
        for q_id in sorted(all_questions):
            row[q_id] = '✓' if q_id in answers else ''
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)

def create_analysis_dataframes(cached_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create analysis and chunks dataframes from database results."""
    try:
        analysis_rows = []
        chunks_rows = []
        
        logger.info(f"Creating dataframes from results: {cached_results}")
        
        # Handle each question's results
        for question_id, result_json in cached_results.items():
            try:
                # Parse the JSON string if needed
                if isinstance(result_json, str):
                    result = json.loads(result_json)
                else:
                    result = result_json
                
                # Helper function to convert lists to strings
                def convert_to_string(value):
                    if isinstance(value, list):
                        return '\n'.join(str(item) for item in value)
                    return str(value) if value is not None else ''
                
                # Create analysis row with string conversions
                analysis_row = {
                    'Question ID': str(question_id),
                    'Analysis': str(result.get('answer', '')),
                    'Score': float(result.get('score', 0)),
                    'Key Evidence': convert_to_string([e.get('text', '') for e in result.get('evidence', [])]),
                    'Gaps': convert_to_string(result.get('gaps', [])),
                    'Sources': convert_to_string(result.get('sources', []))
                }
                analysis_rows.append(analysis_row)
                
                # Process chunks
                chunks = result.get('chunks', result.get('CHUNKS', []))
                evidence_list = result.get('evidence', result.get('EVIDENCE', []))
                evidence_texts = [str(e.get('text', '')) for e in evidence_list]
                
                for i, chunk in enumerate(chunks):
                    chunk_text = str(chunk.get('text', ''))
                    chunk_row = {
                        'Question ID': str(question_id),
                        'Chunk Text': chunk_text,
                        'Vector Similarity': float(chunk.get('similarity', chunk.get('SIMILARITY', 0.0))),
                        'LLM Score': float(chunk.get('llm_score', chunk.get('LLM_SCORE', 0.0))),
                        'Evidence Reference': bool(any(chunk_text in evidence_text for evidence_text in evidence_texts)),
                        'Position': i + 1
                    }
                    chunks_rows.append(chunk_row)
                    
            except Exception as e:
                logger.error(f"Error processing result for question {question_id}: {str(e)}")
                logger.error(f"Result data: {result_json}")
                continue
        
        # Create DataFrames
        analysis_df = pd.DataFrame(analysis_rows) if analysis_rows else pd.DataFrame()
        chunks_df = pd.DataFrame(chunks_rows) if chunks_rows else pd.DataFrame()
        
        # Ensure all object columns are strings
        for df in [analysis_df, chunks_df]:
            if not df.empty:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str)
        
        logger.info(f"Created dataframes - Analysis: {len(analysis_rows)} rows, Chunks: {len(chunks_rows)} rows")
        logger.info(f"Analysis columns dtypes: {analysis_df.dtypes.to_dict() if not analysis_df.empty else 'No data'}")
        
        return analysis_df, chunks_df
        
    except Exception as e:
        logger.error(f"Error creating dataframes: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def display_analysis_results(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame, file_key: str = None) -> None:
    """Display analysis results in a consistent format for both individual and consolidated views"""
    try:
        if analysis_df.empty:
            st.warning("No analysis results to display")
            return
            
        # Analysis Results Table
        st.subheader("Analysis Results")
        st.dataframe(
            data=analysis_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Question ID": st.column_config.TextColumn(
                    "Question ID",
                    width="small",
                ),
                "Analysis": st.column_config.TextColumn(
                    "Analysis",
                    width="large",
                ),
                "Score": st.column_config.NumberColumn(
                    "Score",
                    min_value=0,
                    max_value=10,
                    format="%.1f",
                    width="small",
                ),
                "Key Evidence": st.column_config.TextColumn(
                    "Key Evidence",
                    width="medium",
                ),
                "Gaps": st.column_config.TextColumn(
                    "Gaps",
                    width="medium",
                ),
                "Sources": st.column_config.TextColumn(
                    "Sources",
                    width="small",
                )
            }
        )
        
        # Document Chunks Table
        if not chunks_df.empty:
            st.subheader("Document Chunks")
            st.dataframe(
                data=filter_dataframe(chunks_df),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Question ID": st.column_config.TextColumn(
                        "Question ID",
                        width="small",
                    ),
                    "Chunk Text": st.column_config.TextColumn(
                        "Text",
                        width="large",
                    ),
                    "Vector Similarity": st.column_config.NumberColumn(
                        "Vector Similarity",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    ),
                    "LLM Score": st.column_config.NumberColumn(
                        "LLM Score",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    ),
                    "Evidence Reference": st.column_config.CheckboxColumn(
                        "Is Evidence",
                    ),
                    "Position": st.column_config.NumberColumn(
                        "Position",
                        width="small",
                    )
                }
            )
            
        # Add download buttons if file_key is provided
        if file_key:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Analysis Results",
                    convert_df(analysis_df),
                    f"analysis_results_{file_key}.csv",
                    "text/csv",
                    key=f"download_analysis_{file_key}"
                )
            
            with col2:
                st.download_button(
                    "Download Chunks",
                    convert_df(chunks_df),
                    f"chunks_{file_key}.csv",
                    "text/csv",
                    key=f"download_chunks_{file_key}"
                )
                
    except Exception as e:
        logger.error(f"Error displaying analysis results: {str(e)}", exc_info=True)
        st.error(f"Error displaying results: {str(e)}")

def display_consolidated_results(question_set: str):
    """Display consolidated results for all reports"""
    try:
        # Initialize analyzer and set question set
        analyzer = ReportAnalyzer()
        analyzer.analyzer.update_question_set(question_set)
        
        # Get all available cache configurations
        cache_configs = analyzer.analyzer.cache_manager.check_cache_status()
        logger.info(f"Found cache configs: {cache_configs}")  # Debug log
        
        if not cache_configs:
            st.warning("No cached analyses found")
            return
            
        # Create a readable format for the config selection
        config_options = []
        for config in cache_configs:
            if len(config) == 6:  # Full config row from cache_status
                file_path, chunk_size, chunk_overlap, top_k, model, qs = config
                if qs == question_set:  # Only show configs for selected question set
                    label = f"File: {Path(file_path).name}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Top-K: {top_k}, Model: {model}"
                    config_options.append({
                        'label': label,
                        'config': {
                            'file_path': file_path,
                            'chunk_size': chunk_size,
                            'chunk_overlap': chunk_overlap,
                            'top_k': top_k,
                            'model': model,
                            'question_set': qs
                        }
                    })
        
        if not config_options:
            st.warning(f"No cached results found for question set: {question_set}")
            return
            
        # Let user select a configuration
        st.subheader("Select Analysis Configuration")
        selected_config = st.selectbox(
            "Choose a configuration",
            options=config_options,
            format_func=lambda x: x['label'],
            key="consolidated_config_select"
        )
        
        if selected_config:
            # Get results for selected configuration
            cached_results = analyzer.analyzer.cache_manager.get_analysis(
                file_path=selected_config['config']['file_path'],
                config=selected_config['config']
            )
            
            if cached_results:
                # Process answers into dataframes
                analysis_df, chunks_df = create_analysis_dataframes(cached_results)
                
                # Display results
                display_analysis_results(analysis_df, chunks_df)
                
                # Add download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Analysis Results",
                        convert_df(analysis_df),
                        f"analysis_results_{question_set}.csv",
                        "text/csv",
                        key=f"download_analysis_{question_set}"
                    )
                
                with col2:
                    st.download_button(
                        "Download Chunks",
                        convert_df(chunks_df),
                        f"chunks_{question_set}.csv",
                        "text/csv",
                        key=f"download_chunks_{question_set}"
                    )
            else:
                st.warning("No results found for selected configuration")
                
    except Exception as e:
        logger.error(f"Error displaying consolidated results: {str(e)}", exc_info=True)
        st.error(f"Error displaying consolidated results: {str(e)}")

def display_cache_selector(file_path: str):
    """Display cache management options"""
    st.subheader("Cache Management")
    
    # Get current configuration
    current_config = {
        'chunk_size': st.session_state.new_chunk_size,
        'chunk_overlap': st.session_state.new_overlap,
        'top_k': st.session_state.new_top_k,
        'model': st.session_state.new_llm_model,
        'question_set': st.session_state.new_question_set
    }
    
    if 'analyzer' in st.session_state:
        # Show cache status using cache manager
        try:
            cache_entries = st.session_state.analyzer.analyzer.cache_manager.check_cache_status(file_path)
            if cache_entries:
                st.text(f"Found {len(cache_entries)} cached configurations:")
                for entry in cache_entries:
                    st.text(f"• Configuration: {entry}")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"Current configuration: {current_config}")
                
                with col2:
                    if st.button("🔄 Clear Cache for File"):
                        try:
                            st.session_state.analyzer.analyzer.cache_manager.clear_cache(file_path)
                            st.success(f"Cache cleared for file.")
                            # Clear results from session state
                            if 'results' in st.session_state:
                                del st.session_state.results
                            if 'analysis_df' in st.session_state:
                                del st.session_state.analysis_df
                            if 'chunks_df' in st.session_state:
                                del st.session_state.chunks_df
                            st.session_state.analysis_complete = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing cache: {str(e)}")
            else:
                st.info("No cached analyses available for this file")
        except Exception as e:
            st.error(f"Error checking cache status: {str(e)}")

def get_current_settings(st) -> dict:
    """Get all current settings from the UI widgets"""
    # Get first question set as default
    default_set = list(question_sets.keys())[0]
    
    return {
        'chunk_size': st.session_state.get('new_chunk_size', 500),
        'overlap': st.session_state.get('new_overlap', 20),
        'top_k': st.session_state.get('new_top_k', 5),
        'llm_model': st.session_state.get('new_llm_model', LLM_MODELS[0]),
        'use_llm_scoring': st.session_state.get('new_llm_scoring', False),
        'batch_scoring': st.session_state.get('new_batch_scoring', True),
        'selected_set': st.session_state.get('new_question_set', default_set)
    }

def update_analyzer_parameters():
    """Update analyzer parameters based on current session state"""
    if 'analyzer' in st.session_state:
        # Log current values before update
        logger.info(f"Updating analyzer parameters:")
        logger.info(f"- Chunk size: {st.session_state.new_chunk_size}")
        logger.info(f"- Overlap: {st.session_state.new_overlap}")
        logger.info(f"- Top K: {st.session_state.new_top_k}")
        logger.info(f"- Model: {st.session_state.new_llm_model}")
        
        # Update analyzer parameters
        st.session_state.analyzer.analyzer.update_parameters(
            chunk_size=st.session_state.new_chunk_size,
            chunk_overlap=st.session_state.new_overlap,
            top_k=st.session_state.new_top_k
        )
        st.session_state.analyzer.analyzer.update_llm_model(st.session_state.new_llm_model)
        
        # Log updated values to verify
        logger.info("Updated analyzer parameters:")
        logger.info(f"- Chunk size: {st.session_state.analyzer.analyzer.chunk_params['chunk_size']}")
        logger.info(f"- Overlap: {st.session_state.analyzer.analyzer.chunk_params['chunk_overlap']}")
        logger.info(f"- Top K: {st.session_state.analyzer.analyzer.chunk_params['top_k']}")
        logger.info(f"- Model: {st.session_state.analyzer.analyzer.llm.model}")

def display_analysis_results2():
    """Display analysis results in tabs"""
    logger.info("Starting display_analysis_results")
    
    if not hasattr(st.session_state, 'analysis_df'):
        logger.warning("No analysis_df in session state")
        st.info("No analysis results to display.")
        return
        
    if st.session_state.analysis_df.empty:
        logger.warning("analysis_df is empty")
        st.info("No analysis results to display.")
        return
        
    logger.info(f"Analysis DataFrame shape: {st.session_state.analysis_df.shape}")
    logger.info(f"Analysis DataFrame contents:\n{st.session_state.analysis_df.to_dict('records')}")
    
    tab1, tab2 = st.tabs(["Analysis Results", "Document Chunks"])
    with tab1:
        # Simply display all results we have
        for _, row in st.session_state.analysis_df.iterrows():
            logger.info(f"Displaying row for question {row['Question ID']}")
            import pdb; pdb.set_trace()
            with st.expander(f"Q{row['Question ID']}:", expanded=True):
                st.markdown(f"**Analysis**: {row['Analysis']}")
                st.markdown(f"**Score**: {row['Score']}")
                if row['Key Evidence']:
                    st.markdown("**Key Evidence**:")
                    st.markdown(row['Key Evidence'])
                if row['Gaps']:
                    st.markdown("**Gaps**:")
                    st.markdown(row['Gaps'])
                if row['Sources']:
                    st.markdown("**Sources**:")
                    st.markdown(row['Sources'])
    
    with tab2:
        if hasattr(st.session_state, 'chunks_df') and not st.session_state.chunks_df.empty:
            logger.info(f"Chunks DataFrame shape: {st.session_state.chunks_df.shape}")
            st.dataframe(
                st.session_state.chunks_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            logger.info("No chunks data to display")
            st.info("No chunk data available")

async def run_analysis(analyzer, file_path: str, selected_questions: list, progress_text):
    """Run the analysis process asynchronously"""
    try:
        # ... existing code ...
        
        # Update display using the new function
        if hasattr(st.session_state, 'analysis_df') and hasattr(st.session_state, 'chunks_df'):
            file_key = generate_file_key(file_path, st)
            display_analysis_results(
                st.session_state.analysis_df,
                st.session_state.chunks_df,
                file_key
            )
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        st.error(f"Error during analysis: {str(e)}")

def inspect_cache_database():
    """Inspect and display the contents of the SQLite cache database"""
    try:
        analyzer = ReportAnalyzer()
        cache_manager = analyzer.analyzer.cache_manager
        
        # Get all cache configurations using existing method
        cache_configs = cache_manager.check_cache_status()
        st.write(f"Database path: {cache_manager.db_path}")
        
        if not cache_configs:
            st.warning("No cached analyses found")
            return
            
        st.write(f"Found {len(cache_configs)} cached configurations")
        
        # Show the raw results including chunks
        st.subheader("Cached Results with Chunks:")
        for config in cache_configs:
            if len(config) == 6:  # Full config row
                file_path, chunk_size, chunk_overlap, top_k, model, question_set = config
                
                # Get analysis using existing cache manager method
                results = cache_manager.get_analysis(
                    file_path=file_path,
                    config={
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'top_k': top_k,
                        'model': model,
                        'question_set': question_set
                    }
                )
                
                st.write(f"\n**Configuration:**")
                st.write(f"- File: {Path(file_path).name}")
                st.write(f"- Settings: Chunk={chunk_size}, Overlap={chunk_overlap}, Top-K={top_k}")
                st.write(f"- Model: {model}")
                st.write(f"- Question Set: {question_set}")
                
                if results:
                    for question_id, result in results.items():
                        st.write(f"\n**Question {question_id}**")
                        try:
                            if isinstance(result, str):
                                result = json.loads(result)
                                
                            st.write("Answer:", result.get('answer', 'No answer'))
                            st.write("Score:", result.get('score', 'No score'))
                            
                            # Display chunks if they exist
                            chunks = result.get('chunks', result.get('CHUNKS', []))
                            if chunks:
                                st.write(f"Found {len(chunks)} chunks:")
                                for i, chunk in enumerate(chunks, 1):
                                    st.write(f"\nChunk {i}:")
                                    st.write("Text:", chunk.get('text', 'No text'))
                                    st.write("Similarity:", chunk.get('similarity', chunk.get('SIMILARITY', 'No similarity')))
                                    st.write("LLM Score:", chunk.get('llm_score', chunk.get('LLM_SCORE', 'No LLM score')))
                            else:
                                st.write("No chunks found in result")
                                
                            # Show the complete raw JSON if needed
                            if st.checkbox(f"Show raw JSON for Question {question_id}", key=f"raw_{file_path}_{question_id}"):
                                st.json(result)
                                
                        except Exception as e:
                            st.error(f"Error processing result for question {question_id}: {str(e)}")
                else:
                    st.warning(f"No results found for this configuration")
                    
    except Exception as e:
        logger.error(f"Error inspecting cache: {str(e)}", exc_info=True)
        st.error(f"Error inspecting cache: {str(e)}")

def main():
    try:
        st.set_page_config(page_title="Report Analyst", layout="wide")
        
        # Initialize analyzer with default question set
        try:
            # Initialize analyzer and store in session state if not already there
            if 'analyzer' not in st.session_state:
                st.session_state.analyzer = ReportAnalyzer()
            analyzer = st.session_state.analyzer  # Use the stored analyzer
            
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            st.exception(e)
            return

        st.title("Report Analyst")
        
        # Settings section - moved below the title
        with st.expander("Analysis Configuration", expanded=True):
            # Question set selection
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_set = st.selectbox(
                    "Select Question Set",
                    options=list(question_sets.keys()),
                    format_func=lambda x: question_sets[x]['name'],
                    key="new_question_set",
                    index=0,  # Ensure a default is selected
                    on_change=update_analyzer_parameters
                )
            
            # Show question set description
            with col2:
                if selected_set in question_sets:
                    st.info(question_sets[selected_set]['description'])
            
            # Update analyzer's question set
            analyzer.analyzer.update_question_set(selected_set)
            
            # Clear results if question set changed
            if ('last_question_set' not in st.session_state or 
                st.session_state.last_question_set != selected_set):
                if 'results' in st.session_state:
                    del st.session_state.results
                st.session_state.last_question_set = selected_set
            
            # LLM settings
            col1, col2, col3 = st.columns(3)
            with col1:
                new_llm_scoring = st.checkbox("Use LLM Scoring", value=False, key="new_llm_scoring")
            with col2:
                new_batch_scoring = st.checkbox("Batch Scoring", value=True, key="new_batch_scoring")
            with col3:
                new_llm_model = st.selectbox(
                    "LLM Model", 
                    options=LLM_MODELS,
                    index=0,  # Ensure a default is selected
                    key="new_llm_model",
                    on_change=update_analyzer_parameters
                )
            
            # Chunking parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                new_chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=500,  # Default value
                    key="new_chunk_size",
                    on_change=update_analyzer_parameters
                )
            with col2:
                new_overlap = st.number_input(
                    "Overlap",
                    min_value=0,
                    max_value=100,
                    value=20,  # Default value
                    key="new_overlap",
                    on_change=update_analyzer_parameters
                )
            with col3:
                new_top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=20,
                    value=5,  # Default value
                    key="new_top_k",
                    on_change=update_analyzer_parameters
                )

        # Create tabs
        file_tab, upload_tab, consolidated_tab = st.tabs([
            "Previous Reports",
            "Upload New",
            "Consolidated Results"
        ])

        # Previous Reports tab
        with file_tab:
            previous_files = get_uploaded_files_history()
            if previous_files:
                selected_file = st.selectbox(
                    "Select a previously analyzed report",
                    options=previous_files,
                    format_func=lambda x: x['name'],
                    key="previous_file"
                )
                if selected_file:
                    file_path = Path(selected_file['path'])
                    if file_path.exists():
                        # Load questions and handle selection
                        question_set_data = analyzer.load_question_set(st.session_state.new_question_set)
                        questions = question_set_data["questions"]
                        
                        if question_set_data["description"]:
                            st.write(question_set_data["description"])
                        
                        # Add question selection UI
                        st.subheader("Select Questions")
                        selected_questions = []
                        for q_id, q_data in questions.items():
                            if st.checkbox(
                                f"{q_id}: {q_data['text']}", 
                                key=f"individual_question_{q_id}"
                            ):
                                selected_questions.append(q_id)
                        
                        # Analysis button and results
                        if st.button("Analyze Selected Questions", key="analyze_button"):
                            if not selected_questions:
                                st.warning("Please select at least one question to analyze.")
                            else:
                                try:
                                    # Get results from cache or run new analysis
                                    cached_results = analyzer.analyzer.cache_manager.get_analysis(
                                        file_path=str(file_path),
                                        config={
                                            'chunk_size': st.session_state.new_chunk_size,
                                            'chunk_overlap': st.session_state.new_overlap,
                                            'top_k': st.session_state.new_top_k,
                                            'model': st.session_state.new_llm_model,
                                            'question_set': st.session_state.new_question_set
                                        }
                                    )
                                    
                                    if cached_results:
                                        # Process answers into dataframes
                                        analysis_df, chunks_df = create_analysis_dataframes(cached_results)
                                        
                                        # Display results using the working function
                                        file_key = Path(file_path).stem
                                        display_analysis_results(analysis_df, chunks_df, file_key)
                                    else:
                                        st.warning("No results found. Running new analysis...")
                                        # Run new analysis if needed
                                        progress_text = st.empty()
                                        progress_text.info("Starting analysis...")
                                        
                                        asyncio.run(run_analysis(
                                            analyzer=analyzer,
                                            file_path=str(file_path),
                                            selected_questions=selected_questions,
                                            progress_text=progress_text
                                        ))
                                        
                                except Exception as e:
                                    logger.error(f"Error during analysis: {str(e)}", exc_info=True)
                                    st.error(f"Error during analysis: {str(e)}")
                    else:
                        st.error(f"File not found: {file_path}")
            else:
                st.info("No previously analyzed reports found")

        # Upload New tab
        with upload_tab:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                if file_path and file_path != st.session_state.get('current_file'):
                    st.session_state.current_file = file_path
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.analysis_complete = False
                    st.session_state.analysis_triggered = False
                    if 'results' in st.session_state:
                        del st.session_state.results
                    st.success(f"File uploaded successfully: {uploaded_file.name}")
                    if not st.session_state.get('file_processed'):
                        st.session_state.file_processed = True
                        st.rerun()

        # Consolidated Results tab
        with consolidated_tab:
            st.header("View All Results")
            st.write("View and export consolidated results for all analyzed reports")
            
            # Add debug button for cache inspection
            if st.button("Inspect Cache Database", key="inspect_cache"):
                inspect_cache_database()
            
            # Question set selection for consolidated view
            selected_set = st.selectbox(
                "Select Question Set",
                options=list(question_sets.keys()),
                format_func=lambda x: question_sets[x]['name'],
                key="consolidated_set"
            )
            
            if selected_set:
                # Show question set description
                if selected_set in question_sets:
                    st.info(question_sets[selected_set]['description'])
                
                # Only show consolidated results
                display_consolidated_results(selected_set)

        # Add Climate+Tech footer at the end
        footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        .footer img {
            height: 30px;
            vertical-align: middle;
            margin-right: 10px;
        }
        </style>
        <div class="footer">
            <img src="https://www.climateandtech.com/climateandtech.png" alt="Climate+Tech Logo">
            <p>Climate+Tech Sustainability Report Analysis Tool</p>
            <p>For custom tool development contact us at <a href="https://www.climateandtech.com" target="_blank">www.climateandtech.com</a></p>
        </div>
        """
        st.markdown(footer, unsafe_allow_html=True)


    except Exception as e:
        st.error("Error during analysis:")
        st.exception(e)

if __name__ == "__main__":
    main() 