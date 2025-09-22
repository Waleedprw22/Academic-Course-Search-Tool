"""
Module for evaluating the Retrieval-Augmented Generation (RAG) system.

This module contains:
- A labeled dataset of queries and their expected course codes for evaluation.
- Functions to perform RAG system evaluation, including making HTTP calls to the backend.
- Calculation of micro-averaged precision and recall metrics across the dataset.
"""

import pandas as pd
from typing import List, Dict, Any
import os
import json
import asyncio
import aiohttp

# Configuration for the API key and backend URL.
# The API_KEY is loaded from environment variables for security, defaulting to a placeholder.
# **IMPORTANT**: Change "YOUR_SECRET_API_KEY" to a strong, unique key in production.
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY") 
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Labeled Dataset for Evaluation ---
# This dataset contains example queries and the 'ground truth' expected course codes.
# It is used to calculate precision and recall of the retrieval system.
LABELED_DATASET = [
    {
        "query": "Who teaches AFRI 0090 and when do they meet?",
        "expected_course_codes": ["AFRI 0090"]
    },
    {
        "query": "Courses related to quantum mechanics at MIT",
        "expected_course_codes": ["8.04", "5.73", "8.370x"] # Example expected courses for quantum mechanics.
    },
    {
        "query": "Introduction to computer science at Brown",
        "expected_course_codes": ["CSCI 0190", "CSCI 0170"] # Example expected courses for intro CS.
    }
]

# --- Direct Evaluation Function (without HTTP calls) ---
# This function is retained for scenarios where direct programmatic access to 
# the RAG components (embedding models, FAISS indices, hybrid_search) is needed 
# without initiating HTTP requests, primarily for internal testing or debugging.
# NOTE: This function requires direct import of `hybrid_search` and access to
# `embedding_models`, `faiss_indices`, and `metadata_dfs` which are typically
# managed by the FastAPI `app.py`'s lifespan events.
def evaluate_rag_system_direct(dataset: List[Dict[str, Any]], embedding_models, faiss_indices, metadata_dfs) -> Dict[str, Any]:
    """
    Performs a direct evaluation of the RAG system by calling `hybrid_search` internally.
    This bypasses HTTP calls and directly interacts with the core search logic.

    Args:
        dataset (List[Dict[str, Any]]): A list of labeled queries and their expected course codes.
        embedding_models: Dictionary of loaded embedding models.
        faiss_indices: Dictionary of loaded FAISS indices for different models and fields.
        metadata_dfs: Dictionary of loaded metadata DataFrames for different models and fields.

    Returns:
        Dict[str, Any]: A dictionary containing micro-averaged precision, recall, 
                        and detailed results for each query.
    """
    from data_processing import hybrid_search # Lazy import to avoid circular dependency.
    
    total_precision = 0.0 # Accumulator for precision for macro-averaging (if used).
    total_recall = 0.0    # Accumulator for recall for macro-averaging (if used).
    num_queries = len(dataset)
    detailed_results = []
    
    model_name = "MiniLM"  # Specify the embedding model to use for direct evaluation.
    if model_name not in embedding_models:
        return {"error": f"Error: Embedding model '{model_name}' not available for direct evaluation."}
    
    model = embedding_models[model_name]
    
    for entry in dataset:
        query = entry["query"]
        expected_courses = set(entry["expected_course_codes"])
        
        try:
            # Generate query embedding using the specified model.
            query_embedding = model.encode(query).tolist()
            
            # Perform hybrid search directly, mimicking the backend's logic.
            retrieved_results_raw = hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                faiss_indices_for_model=faiss_indices[model_name],
                metadata_dfs_for_model=metadata_dfs[model_name],
                model=model,
                k=10, # Retrieve a reasonable number of courses for evaluation.
                alpha=0.2, # Use a typical alpha value for hybrid search.
                filters=None # No filters applied during this direct evaluation.
            )
            
            # Extract course codes from the retrieved results.
            retrieved_course_codes = set()
            for course_data, score in retrieved_results_raw:
                retrieved_course_codes.add(course_data["course_code"])
            
            # Calculate Precision and Recall for the current query (macro-averaged components).
            true_positives = len(expected_courses.intersection(retrieved_course_codes))
            
            precision = true_positives / len(retrieved_course_codes) if len(retrieved_course_codes) > 0 else 0
            recall = true_positives / len(expected_courses) if len(expected_courses) > 0 else 0
            
            total_precision += precision # Accumulate for macro-averaging.
            total_recall += recall       # Accumulate for macro-averaging.
            
            detailed_results.append({
                "query": query,
                "expected_courses": list(expected_courses),
                "retrieved_courses": list(retrieved_course_codes),
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall
            })
            
        except Exception as e:
            # Log errors during evaluation to aid in debugging.
            print(f"Error evaluating query '{query}' (direct evaluation): {e}")
            detailed_results.append({
                "query": query, 
                "error": str(e),
                "expected_courses": list(expected_courses),
                "retrieved_courses": [],
                "precision": 0,
                "recall": 0
            })
    
    # Calculate macro-averaged precision and recall.
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    
    return {
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "num_queries_evaluated": num_queries,
        "detailed_results": detailed_results
    }

# --- HTTP-based Evaluation Function (for external testing) ---
async def evaluate_rag_system_http(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs RAG system evaluation by making asynchronous HTTP POST requests to the
    FastAPI backend's `/query` endpoint for each entry in the labeled dataset.
    Calculates micro-averaged precision and recall based on the retrieved course codes.

    Args:
        dataset (List[Dict[str, Any]]): A list of labeled queries and their expected course codes.

    Returns:
        Dict[str, Any]: A dictionary containing micro-averaged precision, recall,
                        total counts, and detailed results for each query.
    """
    total_true_positives = 0 # Accumulator for micro-averaged true positives.
    total_retrieved_items = 0 # Accumulator for total items retrieved.
    total_expected_items = 0 # Accumulator for total expected items.
    num_queries = len(dataset)
    detailed_results = [] # Stores results for individual queries.
    
    headers = {"X-API-Key": API_KEY} # Authentication header for API requests.
    print(f"DEBUG: Using API_KEY for evaluation: {API_KEY}") # For debugging API Key usage.
    
    # Use aiohttp.ClientSession for efficient asynchronous HTTP requests.
    async with aiohttp.ClientSession() as session:
        for entry in dataset:
            query = entry["query"]
            expected_courses = set(entry["expected_course_codes"]) # Convert to set for efficient intersection.
            
            try:
                # Construct the payload for the /query endpoint.
                payload = {
                    "question": query,
                    "top_k": 3, # Evaluation at top K=3, adjusted for better precision assessment.
                    "alpha": 0.2, # Consistent alpha value as used in the frontend.
                    "embedding_model": "MiniLM", # Specify the embedding model.
                    "filters": None # No specific filters applied during evaluation.
                }
                
                # Make an asynchronous POST request with a timeout.
                async with session.post(
                    f"{BACKEND_URL}/query", 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90) # Extended timeout for potentially long RAG responses.
                ) as response:
                    response.raise_for_status() # Raise an exception for HTTP errors.
                    result = await response.json()
                    
                    retrieved_course_codes = set()
                    if result.get("retrieved_courses"):
                        for course in result["retrieved_courses"]:
                            retrieved_course_codes.add(course["course_code"]) # Collect retrieved course codes.
                    
                    # Calculate True Positives for this query by finding intersection with expected courses.
                    true_positives = len(expected_courses.intersection(retrieved_course_codes))
                    
                    # Update global counts for micro-averaging across the entire dataset.
                    total_true_positives += true_positives
                    total_retrieved_items += len(retrieved_course_codes)
                    total_expected_items += len(expected_courses)

                    # Store detailed results for the current query.
                    detailed_results.append({
                        "query": query,
                        "expected_courses": list(expected_courses),
                        "retrieved_courses": list(retrieved_course_codes),
                        "true_positives": true_positives,
                        "retrieved_count": len(retrieved_course_codes),
                        "expected_count": len(expected_courses),
                    })
                    
            except Exception as e:
                # Log errors during HTTP calls or response processing.
                print(f"Error evaluating query '{query}': {e}")
                detailed_results.append({
                    "query": query, 
                    "error": str(e),
                    "status_code": response.status if 'response' in locals() else "N/A" # Include HTTP status if available.
                })

    # Calculate micro-averaged precision and recall for the entire dataset.
    # Micro-averaging sums up true positives, retrieved items, and expected items across all queries
    # before calculating the final precision and recall, giving equal weight to each item.
    avg_precision = total_true_positives / total_retrieved_items if total_retrieved_items > 0 else 0
    avg_recall = total_true_positives / total_expected_items if total_expected_items > 0 else 0

    return {
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "total_true_positives": total_true_positives,
        "total_retrieved_items": total_retrieved_items,
        "total_expected_items": total_expected_items,
        "num_queries_evaluated": num_queries,
        "detailed_results": detailed_results
    }

# Backward compatibility - keep the old function name for the `/evaluate` endpoint in `app.py`.
async def evaluate_rag_system(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wrapper function for backward compatibility, routing to `evaluate_rag_system_http`.

    Args:
        dataset (List[Dict[str, Any]]): The labeled dataset for evaluation.

    Returns:
        Dict[str, Any]: Evaluation results from `evaluate_rag_system_http`.
    """
    return await evaluate_rag_system_http(dataset)

if __name__ == "__main__":
    """
    Main execution block for running the RAG system evaluation directly.
    This uses asyncio to run the asynchronous HTTP-based evaluation.
    """
    print("Running RAG system evaluation...")
    # Run the asynchronous evaluation function.
    results = asyncio.run(evaluate_rag_system_http(LABELED_DATASET))
    print("\n--- Overall Evaluation Results ---")
    print(json.dumps(results, indent=2))