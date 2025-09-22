# Academic Course Search RAG API

This project implements an academic course search tool using a Retrieval-Augmented Generation (RAG) architecture. It leverages embedding models, FAISS for efficient vector search, keyword matching, and a Large Language Model (LLM) (Groq AI) to provide intelligent answers to user queries about academic courses. A Streamlit frontend offers an intuitive user interface with filtering capabilities.

## Features

-   **Hybrid Search**: Combines semantic vector similarity search (using Sentence Transformers and FAISS) with keyword matching for robust retrieval.
-   **Multi-Field Embeddings**: Course data (title, description, prerequisites) is embedded separately for more granular relevance matching.
-   **Flexible Filtering**: Users can filter search results by institute, department, instructor, meeting times, and prerequisites.
-   **Generative AI Responses**: Uses Groq AI to generate natural language answers based on retrieved course context.
-   **Streamlit Frontend**: A user-friendly web interface for asking queries and applying filters.
-   **API Endpoints**: FastAPI backend providing `/query` and `/evaluate` endpoints.
-   **Basic Token Authentication**: Secure API access using an `X-API-Key` header.
-   **Comprehensive Logging**: Detailed logging of queries, retrieval processes, scores, and response times for monitoring and debugging.
-   **Precision/Recall Evaluation**: An integrated mechanism to evaluate retrieval performance (micro-averaged precision and recall) against a labeled dataset.

## Setup

Follow these steps to set up and run the application locally.

### 1. Clone the Repository

```bash
git clone <repository_url> # Replace with your repository URL
cd Academic-Course-Search-Tool
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .rag
source .rag/bin/activate  # On Windows, use `.\.rag\Scripts\activate`
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

The application requires API keys for Groq AI and for internal API authentication.

-   **`GROQ_API_KEY`**: Obtain this from [Groq Cloud](https://console.groq.com/). You can set it as an environment variable or directly in `app.py` (though environment variables are recommended for production).
-   **`API_KEY`**: This is for internal authentication of your FastAPI endpoints. **Change `YOUR_SECRET_API_KEY` in `app.py` and `evaluation.py` (and potentially `frontend.py` if hardcoded for local testing) to a strong, unique secret key.** For production, load this from an environment variable.

    Example environment variable setup (for Linux/macOS):
    ```bash
    export GROQ_API_KEY="gsk_YOUR_GROQ_API_KEY"
    export API_KEY="YOUR_STRONG_SECRET_API_KEY"
    ```

### 5. Generate FAISS Indices and Metadata

This is a crucial step. The `data_processing.py` script preprocesses your course data, creates embeddings for different fields (title, description, prerequisites), and builds the FAISS indices that the search API uses.

```bash
python data_processing.py
```

This script will output messages indicating its progress and confirm when indices and metadata files (e.g., `faiss_index_MiniLM_title.faiss`, `course_metadata_MiniLM_description.json`) have been saved. Ensure this step completes without errors.

## Running the Application

The application consists of a FastAPI backend and a Streamlit frontend. Both need to be running concurrently.

### 1. Start the FastAPI Backend

Open your terminal, navigate to the project root, activate your virtual environment, and run the backend. **It's important to use a sufficient timeout, especially for the evaluation endpoint.**

```bash
source .rag/bin/activate
uvicorn app:app --reload --timeout 300
```

The `--timeout 300` flag sets the server timeout to 300 seconds (5 minutes), which is helpful for long-running evaluation requests.

### 2. Start the Streamlit Frontend

Open a **new terminal window**, navigate to the project root, activate your virtual environment, and run the frontend:

```bash
source .rag/bin/activate
streamlit run frontend.py
```

This will open the Streamlit application in your web browser.

## Usage

### Frontend (Streamlit)

Access the Streamlit application via your browser (usually `http://localhost:8501`).

-   **Ask a Question**: Type your course query into the text area.
-   **Filters**: Use the sidebar controls to filter by Institute, Department, Instructor, Meeting Times, or Prerequisites.
-   **Search Courses**: Click this button to get RAG-generated answers and a list of retrieved courses.
-   **Evaluate RAG System**: Click this button in the sidebar to run the full evaluation pipeline and see precision/recall metrics.

### Backend API (Directly)

You can interact with the FastAPI backend directly using tools like `curl`, Postman, or a custom script. Remember to include the `X-API-Key` header for authenticated endpoints.

-   **Health Check**: `GET /health` (no authentication needed)
-   **Query Endpoint**: `POST /query` (requires `X-API-Key`)
    Example `curl` (replace `YOUR_SECRET_API_KEY`):
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/query' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -H 'X-API-Key: YOUR_SECRET_API_KEY' \
      -d '{
        "question": "What are the prerequisites for AFRI 0090?",
        "top_k": 5,
        "alpha": 0.2,
        "embedding_model": "MiniLM",
        "filters": {"institute": "Brown University"}
      }'
    ```
-   **Evaluate Endpoint**: `POST /evaluate` (requires `X-API-Key`)
    Example `curl` (replace `YOUR_SECRET_API_KEY`):
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/evaluate' \
      -H 'accept: application/json' \
      -H 'X-API-Key: YOUR_SECRET_API_KEY'
    ```

## Evaluation Details

The evaluation process calculates micro-averaged precision and recall over a predefined `LABELED_DATASET` in `evaluation.py`.

-   **`evaluation.py`**: Contains the `LABELED_DATASET` (a list of queries with their expected relevant course codes) and the `evaluate_rag_system` function.
    -   **Expand `LABELED_DATASET`**: For more robust evaluation, you should significantly expand this dataset with diverse queries and comprehensive lists of truly relevant `expected_course_codes`.
    -   **Top-K for Evaluation**: The `top_k` value used during evaluation (`payload[\"top_k\"]` in `evaluate_rag_system`) directly impacts precision and recall. Adjust this value in `evaluation.py` to analyze performance at different retrieval depths.

## Troubleshooting

-   **`ModuleNotFoundError: No module named 'aiohttp'`**: Run `pip install aiohttp`.
-   **`KeyError: 'MiniLM'`**: Ensure `data_processing.py` has been run successfully to generate all multi-field FAISS indices, and restart the FastAPI backend.
-   **`Read timed out`**: Increase the Uvicorn server timeout when starting the backend (e.g., `uvicorn app:app --reload --timeout 300`).
-   **Incorrect Evaluation Metrics**: Review your `LABELED_DATASET` to ensure `expected_course_codes` are accurate and comprehensive for each query. Adjust `top_k` in `evaluation.py` and `alpha` in `frontend.py` (and potentially in `evaluation.py`) to fine-tune retrieval balance.
