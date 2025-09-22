import pandas as pd
from typing import List, Dict, Any
import os
import json
import asyncio
import aiohttp

# Assuming the API key is the same as used by the frontend or from environment
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY") # **CHANGE THIS IN PRODUCTION**
BACKEND_URL = "http://127.0.0.1:8000"

# --- Labeled Dataset for Evaluation ---
LABELED_DATASET = [
    {
        "query": "Who teaches AFRI 0090 and when do they meet?",
        "expected_course_codes": ["AFRI 0090"]
    },
    {
        "query": "Courses related to quantum mechanics at MIT",
        "expected_course_codes": ["8.04", "5.73", "8.370x"] 
    },
    {
        "query": "Introduction to computer science at Brown",
        "expected_course_codes": ["CSCI 0190", "CSCI 0170"]
    }
]

# --- Direct Evaluation Function (without HTTP calls) ---
def evaluate_rag_system_direct(dataset: List[Dict[str, Any]], embedding_models, faiss_indices, metadata_dfs) -> Dict[str, Any]:
    """
    Direct evaluation that doesn't make HTTP calls to avoid circular dependencies
    """
    from data_processing import hybrid_search
    
    total_precision = 0.0
    total_recall = 0.0
    num_queries = len(dataset)
    detailed_results = []
    
    model_name = "MiniLM"  # Use default model
    if model_name not in embedding_models:
        return {"error": f"Model {model_name} not available"}
    
    model = embedding_models[model_name]
    
    for entry in dataset:
        query = entry["query"]
        expected_courses = set(entry["expected_course_codes"])
        
        try:
            # Embed the query directly
            query_embedding = model.encode(query).tolist()
            
            # Perform hybrid search directly
            retrieved_results_raw = hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                faiss_indices_for_model=faiss_indices[model_name],
                metadata_dfs_for_model=metadata_dfs[model_name],
                model=model,
                k=10,
                alpha=0.2,
                filters=None
            )
            
            retrieved_course_codes = set()
            for course_data, score in retrieved_results_raw:
                retrieved_course_codes.add(course_data["course_code"])
            
            # Calculate Precision and Recall for this query
            true_positives = len(expected_courses.intersection(retrieved_course_codes))
            
            precision = true_positives / len(retrieved_course_codes) if len(retrieved_course_codes) > 0 else 0
            recall = true_positives / len(expected_courses) if len(expected_courses) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            
            detailed_results.append({
                "query": query,
                "expected_courses": list(expected_courses),
                "retrieved_courses": list(retrieved_course_codes),
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall
            })
            
        except Exception as e:
            print(f"Error evaluating query '{query}': {e}")
            detailed_results.append({
                "query": query, 
                "error": str(e),
                "expected_courses": list(expected_courses),
                "retrieved_courses": [],
                "precision": 0,
                "recall": 0
            })
    
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
    HTTP-based evaluation using aiohttp for better async support
    """
    total_true_positives = 0
    total_retrieved_items = 0
    total_expected_items = 0
    num_queries = len(dataset)
    detailed_results = []
    
    headers = {"X-API-Key": API_KEY}
    print(f"DEBUG: Using API_KEY for evaluation: {API_KEY}") # Debugging API Key
    
    async with aiohttp.ClientSession() as session:
        for entry in dataset:
            query = entry["query"]
            expected_courses = set(entry["expected_course_codes"])
            
            try:
                payload = {
                    "question": query,
                    "top_k": 3, # Changed from 10 to 5 for evaluation
                    "alpha": 0.2, # Use default alpha from frontend
                    "embedding_model": "MiniLM",
                    "filters": None
                }
                
                async with session.post(
                    f"{BACKEND_URL}/query", 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    retrieved_course_codes = set()
                    if result.get("retrieved_courses"):
                        for course in result["retrieved_courses"]:
                            retrieved_course_codes.add(course["course_code"])
                    
                    # Calculate True Positives for this query
                    true_positives = len(expected_courses.intersection(retrieved_course_codes))
                    
                    # Update global counts for micro-averaging
                    total_true_positives += true_positives
                    total_retrieved_items += len(retrieved_course_codes)
                    total_expected_items += len(expected_courses)

                    detailed_results.append({
                        "query": query,
                        "expected_courses": list(expected_courses),
                        "retrieved_courses": list(retrieved_course_codes),
                        "true_positives": true_positives,
                        "retrieved_count": len(retrieved_course_codes),
                        "expected_count": len(expected_courses),
                    })
                    
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
                detailed_results.append({"query": query, "error": str(e)})

    # Calculate micro-averaged precision and recall for the entire dataset
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

# Backward compatibility - keep the old function name
async def evaluate_rag_system(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wrapper function for backward compatibility
    """
    return await evaluate_rag_system_http(dataset)

if __name__ == "__main__":
    # Example of how to run the evaluation directly
    print("Running RAG system evaluation...")
    results = asyncio.run(evaluate_rag_system_http(LABELED_DATASET))
    print(json.dumps(results, indent=2))