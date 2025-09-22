"""
FastAPI application for Academic Course Search.

This API provides endpoints for searching academic courses using a Retrieval-Augmented Generation (RAG) system.
It integrates embedding models for semantic search, FAISS for efficient vector indexing,
keyword matching, and Groq AI for generating natural language responses.
The application also includes basic token authentication and a mechanism for evaluating
retrieval performance.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import pandas as pd # Added missing import for pandas

# Import functions from data_processing.py and evaluation.py
from data_processing import (
    load_faiss_index_and_metadata,
    initialize_embedding_models,
    hybrid_search,
    generate_rag_response
)
from evaluation import LABELED_DATASET, evaluate_rag_system

# Configure logging for the application.
# Set level to DEBUG to see detailed timing and process information.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store initialized embedding models, FAISS indices, and metadata DataFrames.
# These are initialized once at application startup using the lifespan event.
embedding_models = {}
faiss_indices: Dict[str, Dict[str, Any]] = {}  # Nested dict: model_name -> field_name -> FAISS index
metadata_dfs: Dict[str, Dict[str, pd.DataFrame]] = {} # Nested dict: model_name -> field_name -> metadata DataFrame


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for application startup and shutdown.
    Initializes embedding models, loads FAISS indices, and loads course metadata
    once when the application starts.
    """
    logger.info("Application startup: Initializing models and FAISS indices...")
    global embedding_models, faiss_indices, metadata_dfs
    
    # Initialize embedding models (e.g., MiniLM, MPNet)
    embedding_models = initialize_embedding_models()

    # Define the fields for which embeddings and FAISS indices were created
    embedded_fields = ['title', 'description', 'prerequisites']

    # Load FAISS index and metadata for each embedding model and each embedded field.
    # The structure faiss_indices[model_name][field_name] and metadata_dfs[model_name][field_name]
    # allows for multi-field, multi-model access.
    for model_name in embedding_models.keys():
        faiss_indices[model_name] = {}
        metadata_dfs[model_name] = {}
        for field in embedded_fields:
            try:
                index, metadata_df = load_faiss_index_and_metadata(
                    embedding_column_suffix=field,
                    model_name=model_name
                )
                faiss_indices[model_name][field] = index
                metadata_dfs[model_name][field] = metadata_df
                logger.info(f"Successfully loaded FAISS index and metadata for {model_name} - {field}.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index or metadata for {model_name} - {field}: {e}")
                # Depending on robustness requirements, the application might need to exit here
                # if critical resources fail to load.
    yield
    # Clean up and release resources if needed upon application shutdown.
    logger.info("Application shutdown: Releasing resources...")


# Initialize the FastAPI application with metadata and the custom lifespan event.
app = FastAPI(
    title="Academic Course Search RAG API",
    description="API for searching academic courses using RAG with Groq AI.",
    lifespan=lifespan
)

# --- Configuration for API Keys ---
# Groq API Key for LLM inference. Loaded from environment or hardcoded (for dev only).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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") # **CHANGE THIS IN PRODUCTION**

# API Key for securing FastAPI endpoints. Loaded from environment or hardcoded (for dev only).
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY") # **CHANGE THIS IN PRODUCTION**
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Dependency to validate the API Key provided in the 'X-API-Key' header.
    Raises HTTPException 401 if the key is invalid or missing.
    """
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key

# --- Pydantic Models for Request and Response Schemas ---
class QueryRequest(BaseModel):
    """Schema for incoming search queries."""
    question: str
    top_k: int = 5
    alpha: float = 0.5 # Weighting for hybrid search (0.0 to 1.0)
    filters: Optional[Dict[str, str]] = None # Optional metadata filters (e.g., {"institute": "MIT OpenCourseWare"})
    embedding_model: str = "MiniLM" # Specifies which embedding model to use (e.g., "MiniLM", "MPNet")

class Course(BaseModel):
    """Schema for a retrieved course, including its details and relevance score."""
    institute: str
    course_code: str
    title: str
    instructor: str
    meeting_times: str
    prerequisites: str
    department: str
    description: str
    score: float

class QueryResponse(BaseModel):
    """Schema for the response to a search query."""
    question: str
    answer: str
    retrieved_courses: List[Course]
    retrieval_time_ms: float
    rag_time_ms: float
    total_response_time_ms: float

class EvaluationResponse(BaseModel):
    """Schema for the response from the evaluation endpoint."""
    average_precision: float
    average_recall: float
    total_true_positives: int
    total_retrieved_items: int
    total_expected_items: int
    num_queries_evaluated: int
    detailed_results: List[Dict[str, Any]]

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns a simple status to indicate if the API is running.
    """
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
async def query_courses(request: QueryRequest):
    """
    Processes a user's course search query.
    Performs hybrid search (vector + keyword) with optional filtering,
    generates a RAG response using Groq AI, and returns relevant courses
    along with timing metrics.
    Requires API key authentication.
    """
    start_total_time = time.perf_counter()
    
    logger.info(f"Received query: {request.question} with filters: {request.filters}")

    model_name = request.embedding_model
    if model_name not in embedding_models:
        raise HTTPException(status_code=400, detail=f"Embedding model '{model_name}' not loaded.")

    model = embedding_models[model_name]
    
    # 1. Embed the query using the specified embedding model.
    query_embedding = model.encode(request.question).tolist()

    # 2. Perform Hybrid Search to retrieve relevant courses.
    start_retrieval_time = time.perf_counter()
    retrieved_results_raw = hybrid_search(
        query_embedding=query_embedding,
        query_text=request.question,
        faiss_indices_for_model=faiss_indices[model_name], # Pass all indices for the model
        metadata_dfs_for_model=metadata_dfs[model_name],   # Pass all metadata dfs for the model
        model=model,
        k=request.top_k,
        alpha=request.alpha,
        filters=request.filters
    )
    retrieval_time_ms = (time.perf_counter() - start_retrieval_time) * 1000

    # Convert raw retrieval results into the Pydantic Course model for consistent API output.
    retrieved_courses_formatted = []
    for course_data, score in retrieved_results_raw:
        retrieved_courses_formatted.append(Course(**course_data, score=score))
    
    # 3. Generate RAG response using Groq AI based on the query and retrieved context.
    start_rag_time = time.perf_counter()
    rag_answer = generate_rag_response(request.question, retrieved_results_raw, GROQ_API_KEY)
    rag_time_ms = (time.perf_counter() - start_rag_time) * 1000

    total_response_time_ms = (time.perf_counter() - start_total_time) * 1000

    # Enhanced logging of query details, retrieved courses, and performance metrics.
    logger.info("--- Query Details ---")
    logger.info(f"Question: {request.question}")
    logger.info(f"Filters: {request.filters}")
    logger.info(f"Embedding Model: {request.embedding_model}")
    logger.info(f"Top K: {request.top_k}, Alpha: {request.alpha}")
    logger.info(f"Retrieval Time: {retrieval_time_ms:.2f}ms")
    logger.info(f"RAG Generation Time: {rag_time_ms:.2f}ms")
    logger.info(f"Total Response Time: {total_response_time_ms:.2f}ms")
    logger.info(f"Retrieved {len(retrieved_courses_formatted)} courses:")
    for course in retrieved_courses_formatted:
        logger.info(f"  - Course Code: {course.course_code}, Title: {course.title}, Score: {course.score:.4f}")
    logger.info(f"Answer (snippet): {rag_answer[:200]}...")
    logger.info("---------------------")

    return QueryResponse(
        question=request.question,
        answer=rag_answer,
        retrieved_courses=retrieved_courses_formatted,
        retrieval_time_ms=retrieval_time_ms,
        rag_time_ms=rag_time_ms,
        total_response_time_ms=total_response_time_ms
    )

@app.post("/evaluate", response_model=EvaluationResponse, dependencies=[Depends(get_api_key)])
async def evaluate_performance():
    """
    Evaluates the RAG system's retrieval performance (precision and recall)
    against a predefined labeled dataset.
    This endpoint triggers the `evaluate_rag_system` function from `evaluation.py`,
    which makes internal HTTP calls to the `/query` endpoint.
    Returns aggregated and detailed evaluation metrics.
    Requires API key authentication.
    """
    logger.info("Received evaluation request.")
    evaluation_results = await evaluate_rag_system(LABELED_DATASET)
    logger.info(f"RAG Evaluation Results: {evaluation_results}")
    return evaluation_results

