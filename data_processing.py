"""
Data processing and core RAG (Retrieval-Augmented Generation) logic for the Academic Course Search tool.

This module handles:
- Loading and combining course data from various sources (e.g., Brown, MIT).
- Initializing and managing Sentence Transformer embedding models.
- Generating multi-field embeddings for course titles, descriptions, and prerequisites.
- Creating, saving, and loading FAISS indices for efficient vector similarity search.
- Implementing hybrid search, combining vector similarity with keyword matching.
- Formatting meeting times for better readability.
- Generating natural language responses using a Groq AI LLM based on retrieved context.
"""

import pandas as pd
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from groq import Groq
import re
from datetime import datetime
import time # Added for timing performance metrics

from brown_data_extraction import get_brown_courses_dataframe
from mit_data_extraction import get_mit_courses_dataframe

# Configure logging for this module. Set to DEBUG for detailed timing information.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_combine_course_data() -> pd.DataFrame:
    """
    Loads course data from Brown University and MIT OpenCourseWare,
    then combines them into a single Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing combined course information.
    """
    logger.info("Loading Brown University course data...")
    brown_df = get_brown_courses_dataframe()
    logger.info(f"Brown University courses loaded: {len(brown_df)}")

    logger.info("Loading MIT OpenCourseWare data...")
    mit_df = get_mit_courses_dataframe()
    logger.info(f"MIT OpenCourseWare courses loaded: {len(mit_df)}")

    combined_df = pd.concat([brown_df, mit_df], ignore_index=True)
    logger.info(f"Total combined courses: {len(combined_df)}")
    return combined_df

def initialize_embedding_models() -> Dict[str, SentenceTransformer]:
    """
    Initializes and returns a dictionary of Sentence Transformer embedding models.

    Returns:
        Dict[str, SentenceTransformer]: A dictionary mapping model names to their instances.
    """
    logger.info("Initializing embedding models...")
    models = {
        "MiniLM": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        "MPNet": SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    }
    logger.info("Embedding models initialized.")
    return models

def embed_course_data(df: pd.DataFrame, model: SentenceTransformer, model_name: str) -> pd.DataFrame:
    """
    Generates embeddings for specified text fields (title, description, prerequisites)
    of the course data using a given embedding model.
    New embedding columns are added to the DataFrame for each field.

    Args:
        df (pd.DataFrame): The input DataFrame containing course data.
        model (SentenceTransformer): The embedding model instance to use.
        model_name (str): The name of the embedding model (e.g., "MiniLM").

    Returns:
        pd.DataFrame: The DataFrame with new embedding columns for each specified field.
    """
    logger.info(f"Embedding course data using {model_name} for individual fields...")
    embedded_fields = ['title', 'description', 'prerequisites']
    for field in embedded_fields:
        # Ensure the field exists and is a string before embedding to avoid errors.
        # Missing or non-string values will result in None for the embedding.
        df[f'embedding_{model_name}_{field}'] = df[field].apply(
            lambda x: model.encode(x).tolist() if isinstance(x, str) and pd.notna(x) else None
        )
    logger.info(f"Embedding complete for {model_name} across specified fields.")
    return df

def create_and_save_faiss_index(df: pd.DataFrame, embedding_column_name: str, model_name: str) -> Optional[Any]:
    """
    Creates a FAISS index for a specified embedding column and saves it to a file.
    Also saves the corresponding course metadata (without embedding columns) to a JSON file.

    Args:
        df (pd.DataFrame): The DataFrame containing course data with embeddings.
        embedding_column_name (str): The name of the column containing the embeddings
                                     for which to create the FAISS index (e.g., "embedding_MiniLM_title").
        model_name (str): The name of the embedding model.

    Returns:
        Optional[Any]: The created FAISS index object, or None if no valid embeddings were found.
    """
    logger.info(f"Creating FAISS index for {embedding_column_name} (using {model_name})...")
    # Filter out rows where the specific embedding column might be None or empty.
    df_filtered = df.dropna(subset=[embedding_column_name])
    
    if df_filtered.empty:
        logger.warning(f"No valid embeddings found for {embedding_column_name}. Skipping index creation.")
        return None

    embeddings = np.array(df_filtered[embedding_column_name].tolist()).astype('float32')
    dimension = embeddings.shape[1]

    # Using IndexFlatL2 for a simple L2 distance index (Euclidean distance).
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Construct file names for index and metadata based on model and field.
    field_suffix = embedding_column_name.split(f"embedding_{model_name}_")[1]
    index_file_name = f'faiss_index_{model_name}_{field_suffix}.faiss'
    metadata_file_name = f'course_metadata_{model_name}_{field_suffix}.json'
    
    faiss.write_index(index, index_file_name)
    logger.info(f"FAISS index saved as {index_file_name}")

    # Save metadata (the DataFrame without *any* embedding columns) to JSON.
    # This ensures the metadata JSON is clean and doesn't store large embedding vectors.
    df_metadata = df_filtered.drop(columns=[col for col in df_filtered.columns if 'embedding_' in col])
    df_metadata.to_json(metadata_file_name, orient="records", indent=2)
    logger.info(f"Metadata saved as {metadata_file_name}")
    
    return index

def load_faiss_index_and_metadata(embedding_column_suffix: str, model_name: str) -> Tuple[Any, pd.DataFrame]:
    """
    Loads a FAISS index and its corresponding metadata DataFrame from files.

    Args:
        embedding_column_suffix (str): The suffix representing the embedded field
                                      (e.g., "title", "description", "prerequisites").
        model_name (str): The name of the embedding model.

    Returns:
        Tuple[Any, pd.DataFrame]: A tuple containing the loaded FAISS index and metadata DataFrame.
    """
    logger.info(f"Loading FAISS index for {embedding_column_suffix}...")
    index_name = f'faiss_index_{model_name}_{embedding_column_suffix}.faiss'
    index = faiss.read_index(index_name)
    logger.info(f"FAISS index {index_name} loaded.")

    logger.info(f"Loading metadata for {embedding_column_suffix}...")
    metadata_name = f'course_metadata_{model_name}_{embedding_column_suffix}.json'
    with open(metadata_name, 'r') as f:
        metadata = json.load(f)
    metadata_df = pd.DataFrame(metadata)
    logger.info(f"Metadata {metadata_name} loaded.")

    return index, metadata_df

def filtered_vector_search(query_embedding: List[float], 
                           faiss_indices_for_model: Dict[str, Any], 
                           metadata_dfs_for_model: Dict[str, pd.DataFrame], 
                           k: int = 5, 
                           filters: Optional[Dict[str, str]] = None) -> List[Tuple[Dict[str, Any], float]]:
    """
    Performs a multi-field vector search across different course data fields (title, description, prerequisites).
    Combines results by taking the best score for each unique course and then applies optional metadata filtering.

    Args:
        query_embedding (List[float]): The embedding of the user's query.
        faiss_indices_for_model (Dict[str, Any]): Dictionary of FAISS indices for different fields for a specific model.
                                                   (e.g., {"title": index_title, "description": index_desc}).
        metadata_dfs_for_model (Dict[str, pd.DataFrame]): Dictionary of metadata DataFrames for different fields.
        k (int): The number of top results to retrieve after filtering.
        filters (Optional[Dict[str, str]]): Optional metadata filters (e.g., {"institute": "MIT OpenCourseWare"}).

    Returns:
        List[Tuple[Dict[str, Any], float]]: A list of tuples, each containing a course's metadata (dict) and its score (float).
    """
    start_time_func = time.perf_counter()
    logger.debug("Starting filtered_vector_search...")

    all_field_results = {} # Stores best score per course_code across all fields
    embedded_fields = ['title', 'description', 'prerequisites']

    for field in embedded_fields:
        start_time_field_search = time.perf_counter()
        # Ensure index and metadata are available for the current field
        if field in faiss_indices_for_model and field in metadata_dfs_for_model:
            index = faiss_indices_for_model[field]
            metadata_df = metadata_dfs_for_model[field]
            
            # Search more items than 'k' to allow for effective filtering and combination later.
            D, I = index.search(np.array([query_embedding]).astype('float32'), k * 20)  

            retrieved_indices = I[0]
            retrieved_scores = D[0]

            for i, score in zip(retrieved_indices, retrieved_scores):
                if i == -1: continue # Skip invalid indices from FAISS
                item_metadata = metadata_df.iloc[i].to_dict()
                course_code = item_metadata.get('course_code')

                if course_code:
                    # Store the best (smallest distance) score for each course_code across all fields.
                    if course_code not in all_field_results or all_field_results[course_code][1] > score:
                        all_field_results[course_code] = (item_metadata, score)
        logger.debug(f"Field search for {field} took {(time.perf_counter() - start_time_field_search) * 1000:.2f}ms")
    
    start_time_combine_sort = time.perf_counter()
    # Convert results to a list and sort by score (distance, smaller is better).
    combined_vector_results_raw = sorted(all_field_results.values(), key=lambda x: x[1])
    logger.debug(f"Combine and sort results took {(time.perf_counter() - start_time_combine_sort) * 1000:.2f}ms")

    start_time_filter_results = time.perf_counter()
    # Apply metadata filters to the combined vector search results.
    filtered_results = []
    for item_metadata, score in combined_vector_results_raw:
        match = True
        if filters:
            for key, value in filters.items():
                # Check if the metadata field exists and matches the filter value.
                if item_metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append((item_metadata, score))
    logger.debug(f"Applying filters took {(time.perf_counter() - start_time_filter_results) * 1000:.2f}ms")
    logger.debug(f"Finished filtered_vector_search in {(time.perf_counter() - start_time_func) * 1000:.2f}ms. Found {len(filtered_results)} results.")
    
    # Return the top 'k' results after all processing.
    return filtered_results[:k]

def keyword_search(query_text: str, metadata_df: pd.DataFrame, k: int = 5, search_fields: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
    """
    Performs a keyword search on specified metadata fields to find courses.
    Prioritizes exact course code matches, then substring matches in other fields.

    Args:
        query_text (str): The user's query string.
        metadata_df (pd.DataFrame): The DataFrame containing course metadata.
        k (int): The number of top results to retrieve.
        search_fields (Optional[List[str]]): List of fields to search within.
                                             Defaults to ['title', 'description', 'course_code', 'department'].

    Returns:
        List[Tuple[Dict[str, Any], float]]: A list of tuples, each containing course metadata and a keyword score.
    """
    if search_fields is None:
        search_fields = ['title', 'description', 'course_code', 'department']

    query_text_lower = query_text.lower()
    
    matching_courses = {} # Stores best keyword score per course_code

    # Phase 1: Prioritize exact course codes directly mentioned in the query string.
    for index, row in metadata_df.iterrows():
        course_code_lower = str(row['course_code']).lower()
        # Check if the course code is present as an exact word or a significant substring in the query.
        if course_code_lower == query_text_lower: # Exact match of query to course_code
            score = 1.0 
        elif course_code_lower in query_text_lower: # Course_code is a substring in the query
            score = 0.9 
        else:
            continue # No direct course code match, move to next course

        # Update score if this match is better than any previous one for this course.
        if row['course_code'] not in matching_courses or matching_courses[row['course_code']][1] < score:
            matching_courses[row['course_code']] = (row.to_dict(), score)
            
    # Phase 2: Perform general keyword search on other fields for courses not yet highly scored.
    # This ensures that less precise, but still relevant, keyword matches are captured.
    for index, row in metadata_df.iterrows():
        for field in search_fields:
            # Only consider fields for courses not yet assigned a top score (1.0 or 0.9 from Phase 1).
            if row['course_code'] not in matching_courses or matching_courses[row['course_code']][1] < 0.9:
                if field in row and isinstance(row[field], str) and query_text_lower in row[field].lower():
                    current_score = matching_courses.get(row['course_code'], (None, 0.0))[1]
                    # Only update if the current keyword match is better than what's already stored (or 0).
                    if current_score < 0.5: # Assign a medium score for general substring match.
                        matching_courses[row['course_code']] = (row.to_dict(), 0.5)
                    break # Move to next course once a match is found in any field for this row.
    
    # Convert dictionary values to a list and return top 'k' results, sorted by score (descending).
    return sorted(matching_courses.values(), key=lambda x: x[1], reverse=True)[:k]

def hybrid_search(query_embedding: List[float], 
                  query_text: str, 
                  faiss_indices_for_model: Dict[str, Any], 
                  metadata_dfs_for_model: Dict[str, pd.DataFrame], 
                  model: SentenceTransformer, 
                  k: int = 5, 
                  alpha: float = 0.5, 
                  filters: Optional[Dict[str, str]] = None) -> List[Tuple[Dict[str, Any], float]]:
    """
    Performs a hybrid search combining multi-field vector similarity and keyword matching.
    Results from both methods are combined and re-ranked based on a weighted alpha score.

    Args:
        query_embedding (List[float]): The embedding of the user's query.
        query_text (str): The original user query string.
        faiss_indices_for_model (Dict[str, Any]): Dictionary of FAISS indices for different fields for a specific model.
        metadata_dfs_for_model (Dict[str, pd.DataFrame]): Dictionary of metadata DataFrames for different fields.
        model (SentenceTransformer): The embedding model instance.
        k (int): The number of top results to retrieve.
        alpha (float): Weighting factor for hybrid search (0.0 for pure keyword, 1.0 for pure vector).
        filters (Optional[Dict[str, str]]): Optional metadata filters.

    Returns:
        List[Tuple[Dict[str, Any], float]]: A list of tuples, each containing a course's metadata (dict) and its combined score.
    """
    start_time_func = time.perf_counter()
    logger.debug("Starting hybrid_search...")

    # Phase 1: Perform Multi-Field Vector Search.
    start_time_vector_search = time.perf_counter()
    vector_results = filtered_vector_search(query_embedding, faiss_indices_for_model, metadata_dfs_for_model, k=k, filters=filters)
    vector_courses = {res[0]['course_code']: (res[0], res[1]) for res in vector_results}
    logger.debug(f"Vector search took {(time.perf_counter() - start_time_vector_search) * 1000:.2f}ms. Found {len(vector_results)} vector results.")

    # Phase 2: Perform Keyword Search.
    # For keyword search, a single representative metadata_df is needed.
    # We can use the metadata_df from any field, as they all contain the core metadata (without embeddings).
    representative_metadata_df = next(iter(metadata_dfs_for_model.values()), None)
    if representative_metadata_df is None:
        logger.error("No metadata DataFrames loaded for keyword search. Returning empty results.")
        return []
    
    start_time_keyword_search = time.perf_counter()
    keyword_results = keyword_search(query_text, representative_metadata_df, k=k)
    keyword_courses = {res[0]['course_code']: (res[0], res[1]) for res in keyword_results}
    logger.debug(f"Keyword search took {(time.perf_counter() - start_time_keyword_search) * 1000:.2f}ms. Found {len(keyword_results)} keyword results.")

    # Phase 3: Combine and re-rank results from vector and keyword searches.
    start_time_combine_scores = time.perf_counter()
    combined_results = {} # Stores course_code -> (metadata, combined_score)

    # Identify all unique course codes found by either search method.
    all_course_codes = set(vector_courses.keys()).union(keyword_courses.keys())

    for course_code in all_course_codes:
        vector_score = vector_courses.get(course_code, (None, float('inf')))[1] # Lower distance is better for vector score
        keyword_score = keyword_courses.get(course_code, (None, 0.0))[1] # Higher score is better for keyword score (0.0 to 1.0)
        
        # Normalize vector score: 1 / (1 + distance) to convert distance into a similarity score (higher is better).
        normalized_vector_score = 1 / (1 + vector_score) if vector_score != float('inf') else 0

        # Normalized keyword score is already between 0 and 1 (1.0 for exact, 0.9 for substring, 0.5 for general keyword).
        normalized_keyword_score = keyword_score

        # Combine scores using alpha weighting to balance vector and keyword relevance.
        combined_score = (alpha * normalized_vector_score) + ((1 - alpha) * normalized_keyword_score)

        # Aggressive boosting for strong keyword matches (exact course code or present as substring).
        # This ensures that explicit mentions of course codes are highly prioritized in the final ranking.
        if keyword_score >= 0.9: # Covers both exact (1.0) and substring (0.9) matches
            combined_score = 1.0 + normalized_vector_score # Boosts to ensure top rank, while still reflecting semantic relevance.

        # Retrieve the most complete metadata for the course (preferring vector search metadata if available).
        metadata = vector_courses.get(course_code, (None, None))[0] or keyword_courses.get(course_code, (None, None))[0]

        if metadata:
            combined_results[course_code] = (metadata, combined_score)
    logger.debug(f"Combine scores and boost took {(time.perf_counter() - start_time_combine_scores) * 1000:.2f}ms")

    start_time_sort_final = time.perf_counter()
    # Sort combined results by combined score (descending) and return the top 'k' results.
    sorted_combined_results = sorted(combined_results.values(), key=lambda x: x[1], reverse=True)
    logger.debug(f"Final sort took {(time.perf_counter() - start_time_sort_final) * 1000:.2f}ms")
    logger.debug(f"Finished hybrid_search in {(time.perf_counter() - start_time_func) * 1000:.2f}ms. Returning {len(sorted_combined_results[:k])} results.")
    return sorted_combined_results[:k]

# Helper function to format meeting times from military time and 'Day X' to human-readable format.
def format_meeting_times(meeting_times_str: str) -> str:
    """
    Converts military time and 'Day X' strings in meeting times to a human-readable
    AM/PM and weekday format (e.g., "Day 1 1300-1420" -> "Monday 1:00 PM-2:20 PM").

    Args:
        meeting_times_str (str): The raw meeting times string from course metadata.

    Returns:
        str: The formatted meeting times string.
    """
    if not meeting_times_str:
        return "Not specified"

    # Mapping for day numbers to weekday names.
    day_map = {
        "Day 1": "Monday", "Day 2": "Tuesday", "Day 3": "Wednesday",
        "Day 4": "Thursday", "Day 5": "Friday", "Day 6": "Saturday", "Day 7": "Sunday"
    }

    parts = meeting_times_str.split('; ')
    formatted_parts = []

    for part in parts:
        # Attempt to parse parts like "Day 1 1300-1420".
        match = re.match(r"(Day \d+) (\d{4})-(\d{4})", part)
        if match:
            day_raw, start_time_raw, end_time_raw = match.groups()
            day_formatted = day_map.get(day_raw, day_raw) # Use mapped day or original if not found.

            # Convert military time (HHMM) to datetime objects for formatting.
            try:
                start_time_obj = datetime.strptime(start_time_raw, "%H%M")
                end_time_obj = datetime.strptime(end_time_raw, "%H%M")
                
                start_time_formatted = start_time_obj.strftime("%I:%M %p").lstrip('0') # e.g., "1:00 PM"
                end_time_formatted = end_time_obj.strftime("%I:%M %p").lstrip('0')

                formatted_parts.append(f"{day_formatted} {start_time_formatted}-{end_time_formatted}")
            except ValueError:
                logger.warning(f"Could not parse time format for part: {part}. Keeping original.")
                formatted_parts.append(part) # Fallback to original if time parsing fails.
        else:
            formatted_parts.append(part) # Keep original part if format not recognized.
    
    return "; ".join(formatted_parts)

def generate_rag_response(query: str, retrieved_context: list, groq_api_key: str) -> str:
    """
    Generates a natural language response using Groq AI based on the user's query
    and the retrieved course context.

    Args:
        query (str): The original user query.
        retrieved_context (list): A list of retrieved course metadata and scores.
        groq_api_key (str): The API key for Groq AI.

    Returns:
        str: The AI-generated natural language answer.
    """
    if not groq_api_key:
        return "Error: Groq API key not provided." # Or raise a more appropriate exception.

    # Initialize Groq client with the API key.
    client = Groq(api_key=groq_api_key) # Use passed API key
    
    # Format the retrieved context for the LLM to provide relevant information.
    context_str = "\n\n".join([
        f"Course Title: {c[0].get('title', 'N/A')}\nInstructor: {c[0].get('instructor', 'N/A')}\nMeeting Times: {format_meeting_times(c[0].get('meeting_times', ''))}\nDescription: {c[0].get('description', 'N/A')}\nPrerequisites: {c[0].get('prerequisites', 'N/A')}\nInstitute: {c[0].get('institute', 'N/A')}\nCourse Code: {c[0].get('course_code', 'N/A')}\nScore: {c[1]:.4f}"
        for c in retrieved_context
    ])

    # Define the system prompt to guide the LLM's behavior.
    system_prompt = (
        "You are an AI assistant for academic course search. "
        "Provide concise and helpful answers to user queries about courses, "
        "drawing information only from the provided context. "
        "If the answer is not in the context, state that you don't have enough information. "
        "When listing courses, include their title, course code, institute, and a brief description. "
        "Also, mention the relevance score for each course." 
    )

    # Construct the user prompt with the query and retrieved context.
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Course Information:\n{context_str}\n\n"
        "Based on the retrieved course information, please answer the user query. "
        "List the most relevant courses first, including their title, course code, institute, description, and score. "
        "Then provide a summary answer to the query, if possible, based ONLY on the provided context."
    )

    try:
        # Call the Groq chat completions API.
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="llama-3.3-70b-versatile", # Ensure this is a currently supported Groq model.
            temperature=0.2, # Controls randomness of the output.
            max_tokens=1024 # Maximum number of tokens in the generated response.
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response from Groq AI: {e}")
        return f"Error generating response from Groq AI: {e}"

if __name__ == "__main__":
    """
    Main execution block for generating and saving FAISS indices and metadata files.
    This script should be run directly to prepare the data for the FastAPI application.
    """
    logger.info("Starting data processing and FAISS index generation...")

    # 1. Load and combine course data from various sources.
    combined_course_df = load_and_combine_course_data()
    logger.info("Data loading and combining complete.")

    # 2. Initialize embedding models.
    embedding_models = initialize_embedding_models()
    logger.info("Embedding model initialization complete.")

    # 3. Embed text fields and create FAISS index for each model and each embedded field.
    # The generated FAISS indices and metadata JSON files are crucial for the search API.
    # faiss_indices_for_model and metadata_dfs_for_model are temporary here for script execution.
    faiss_indices_for_model_temp = {}
    metadata_dfs_for_model_temp = {}
    for model_name, model_instance in embedding_models.items():
        logger.info(f"--- Processing for model: {model_name} ---")
        df_with_embeddings = embed_course_data(combined_course_df.copy(), model_instance, model_name)
        logger.info(f"Embedding for {model_name} complete. DataFrame shape with embeddings: {df_with_embeddings.shape}")
        
        # Create FAISS index for each field (title, description, prerequisites).
        # Store results temporarily; the files are what app.py loads.
        faiss_indices_for_model_temp[f'embedding_{model_name}_title'] = create_and_save_faiss_index(df_with_embeddings, f'embedding_{model_name}_title', model_name)
        faiss_indices_for_model_temp[f'embedding_{model_name}_description'] = create_and_save_faiss_index(df_with_embeddings, f'embedding_{model_name}_description', model_name)
        faiss_indices_for_model_temp[f'embedding_{model_name}_prerequisites'] = create_and_save_faiss_index(df_with_embeddings, f'embedding_{model_name}_prerequisites', model_name)
        
        # Load metadata for each field. This step is mainly to verify file existence and structure
        # after saving, as app.py will perform the actual loading.
        metadata_dfs_for_model_temp['title'] = load_faiss_index_and_metadata('title', model_name)[1]
        metadata_dfs_for_model_temp['description'] = load_faiss_index_and_metadata('description', model_name)[1]
        metadata_dfs_for_model_temp['prerequisites'] = load_faiss_index_and_metadata('prerequisites', model_name)[1]

        logger.info(f"FAISS index and metadata for {model_name} saved.")
    logger.info("All models processed and indexes saved. Data processing complete.")