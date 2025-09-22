"""
Streamlit frontend for the Academic Course Search RAG API.

This application provides a user interface to:
- Input natural language queries for academic courses.
- Apply various filters (institute, department, instructor, meeting times, prerequisites).
- Display retrieved courses with their metadata and relevance scores.
- Show a RAG-generated natural language answer based on the retrieved context.
- Trigger an evaluation of the RAG system's precision and recall.
"""

import streamlit as st
import requests
import pandas as pd

# Configuration for the backend API and API Key.
# Ensure the BACKEND_URL matches your FastAPI backend's address.
# The API_KEY must match the one configured in app.py for authentication.
BACKEND_URL = "http://127.0.0.1:8000"
API_KEY = "YOUR_SECRET_API_KEY" # TODO: In a production environment, use Streamlit secrets or environment variables for API keys. **CHANGE THIS IN PRODUCTION**

# Set Streamlit page configuration for a wider layout and custom page title.
st.set_page_config(layout="wide", page_title="Academic Course Search")

st.title("Academic Course Search")
st.markdown("Ask a question to find relevant academic courses.")

# User Input Section
# Text area for the user to type their natural language query.
query = st.text_area("Your Question", help="Enter your question about academic courses, e.g., 'Introduction to Python courses at MIT'.")

# Sidebar for Filters and Configuration
st.sidebar.header("Filters")

# Institute filter: Allows selection from predefined institutes.
selected_institute = st.sidebar.selectbox(
    "Institute",
    options=["All", "MIT OpenCourseWare", "Brown University"],
    index=0, # Default to "All"
    help="Filter courses by the institution they belong to."
)

# Text input filters for various course metadata.
selected_department = st.sidebar.text_input("Department (e.g., Physics)", "", help="Filter courses by department.")
selected_instructor = st.sidebar.text_input("Instructor (e.g., Prof. John Doe)", "", help="Filter courses by instructor's name.")
selected_meeting_times = st.sidebar.text_input("Meeting Times (e.g., MWF 10-11am)", "", help="Filter courses by meeting days/times.")
selected_prerequisites = st.sidebar.text_input("Prerequisites (e.g., Calculus I)", "", help="Filter courses by their prerequisites.")

# Sliders for search parameters: top_k (number of results) and alpha (hybrid search weight).
top_k = st.sidebar.slider("Number of Courses to Retrieve (Top K)", min_value=1, max_value=10, value=5, help="Controls how many top courses are retrieved by the search algorithm.")
alpha = st.sidebar.slider("Hybrid Search Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Adjusts the balance between vector similarity (higher alpha) and keyword matching (lower alpha).")

# Build a dictionary of active filters to send to the backend.
filters = {}
if selected_institute != "All":
    filters["institute"] = selected_institute

if selected_department:
    filters["department"] = selected_department

if selected_instructor:
    filters["instructor"] = selected_instructor

if selected_meeting_times:
    filters["meeting_times"] = selected_meeting_times

if selected_prerequisites:
    filters["prerequisites"] = selected_prerequisites

# Search Button and Results Display Section
if st.button("Search Courses"):
    # Ensure a query is provided before attempting to search.
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            # Show a spinner while the search is in progress.
            with st.spinner("Searching..."):
                # Construct the payload for the /query API endpoint.
                payload = {
                    "question": query,
                    "top_k": top_k,
                    "alpha": alpha,
                    "embedding_model": "MiniLM", # Specify the embedding model to use (e.g., "MiniLM" or "MPNet").
                    "filters": filters if filters else None # Send filters only if they are not empty.
                }

                # Include the API key in the request headers for authentication.
                headers = {
                    "X-API-Key": API_KEY
                }
                
                # Make a POST request to the backend's /query endpoint.
                response = requests.post(f"{BACKEND_URL}/query", json=payload, headers=headers)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx responses).
                result = response.json()

                # Display the RAG-generated answer.
                st.subheader("Generated Answer")
                st.write(result.get("answer", "No answer generated.")) # Fallback message if no answer.
                
                # Display retrieved courses in a DataFrame.
                st.subheader("Retrieved Courses")
                if result.get("retrieved_courses"):
                    courses_df = pd.DataFrame(result["retrieved_courses"])
                    # Define and reorder columns for better presentation in the Streamlit DataFrame.
                    display_columns = [
                        "title", "course_code", "institute", "department", 
                        "instructor", "meeting_times", "prerequisites", "description", "score"
                    ]
                    st.dataframe(courses_df[display_columns])
                else:
                    st.info("No courses retrieved based on your query and filters.")

                # Display performance metrics.
                st.markdown(f"**Retrieval Time:** {result.get("retrieval_time_ms", 0):.2f} ms")
                st.markdown(f"**RAG Generation Time:** {result.get("rag_time_ms", 0):.2f} ms")
                st.markdown(f"**Total Response Time:** {result.get("total_response_time_ms", 0):.2f} ms")

        # Error handling for network and API-related issues.
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the backend is running and accessible.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while communicating with the API: {e}. Please check the backend logs for more details.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}. Please report this issue.")

# RAG System Evaluation Section in Sidebar
st.sidebar.markdown("--- ")
st.sidebar.header("RAG System Evaluation")
if st.sidebar.button("Evaluate RAG System"):
    try:
        with st.spinner("Running evaluation..."):
            # Include API key in headers for the /evaluate endpoint.
            headers = {
                "X-API-Key": API_KEY
            }
            # Make a POST request to the backend's /evaluate endpoint.
            response = requests.post(f"{BACKEND_URL}/evaluate", headers=headers)
            response.raise_for_status() # Raise an exception for HTTP errors.
            evaluation_results = response.json()

            # Display overall evaluation metrics.
            st.subheader("Evaluation Results")
            st.metric(label="Average Precision", value=f"{evaluation_results['average_precision']:.2f}", help="Micro-averaged precision across all queries.")
            st.metric(label="Average Recall", value=f"{evaluation_results['average_recall']:.2f}", help="Micro-averaged recall across all queries.")

            # Display detailed results for each query in the labeled dataset.
            st.markdown("### Detailed Results per Query")
            for detail in evaluation_results['detailed_results']:
                st.json(detail)

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the backend is running and accessible.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during evaluation: {e}. Please check the backend logs for more details.")
    except Exception as e:
        st.error(f"An unexpected error occurred during evaluation: {e}. Please report this issue.")
