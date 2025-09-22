
import streamlit as st
import requests
import pandas as pd

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"
API_KEY = "YOUR_SECRET_API_KEY" # Must match the key in app.py. **CHANGE THIS IN PRODUCTION**

st.set_page_config(layout="wide", page_title="Academic Course Search")

st.title("Academic Course Search")
st.markdown("Ask a question to find relevant academic courses.")

# User Input
query = st.text_area("Your Question")

# Filters
st.sidebar.header("Filters")
selected_institute = st.sidebar.selectbox(
    "Institute",
    options=["All", "MIT OpenCourseWare", "Brown University"],
    index=0
)

selected_department = st.sidebar.text_input("Department (e.g., Physics)", "")

selected_instructor = st.sidebar.text_input("Instructor (e.g., Prof. John Doe)", "")
selected_meeting_times = st.sidebar.text_input("Meeting Times (e.g., MWF 10-11am)", "")
selected_prerequisites = st.sidebar.text_input("Prerequisites (e.g., Calculus I)", "")

top_k = st.sidebar.slider("Number of Courses to Retrieve (Top K)", min_value=1, max_value=10, value=5)
alpha = st.sidebar.slider("Hybrid Search Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Build filters dictionary
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

# Search Button
if st.button("Search Courses"):
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching..."):
                payload = {
                    "question": query,
                    "top_k": top_k,
                    "alpha": alpha,
                    "embedding_model": "MiniLM", # Or "MPNet" if you want to experiment
                    "filters": filters if filters else None
                }

                headers = {
                    "X-API-Key": API_KEY
                }
                
                response = requests.post(f"{BACKEND_URL}/query", json=payload, headers=headers)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                result = response.json()

                st.subheader("Generated Answer")
                st.write(result.get("answer", "No answer generated."))
                
                st.subheader("Retrieved Courses")
                if result.get("retrieved_courses"):
                    courses_df = pd.DataFrame(result["retrieved_courses"])
                    # Reorder columns for better display
                    display_columns = [
                        "title", "course_code", "institute", "department", 
                        "instructor", "meeting_times", "prerequisites", "description", "score"
                    ]
                    st.dataframe(courses_df[display_columns])
                else:
                    st.info("No courses retrieved based on your query and filters.")

                st.markdown(f"**Retrieval Time:** {result.get("retrieval_time_ms", 0):.2f} ms")
                st.markdown(f"**RAG Generation Time:** {result.get("rag_time_ms", 0):.2f} ms")
                st.markdown(f"**Total Response Time:** {result.get("total_response_time_ms", 0):.2f} ms")

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the backend is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while communicating with the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("--- ")
st.sidebar.header("RAG System Evaluation")
if st.sidebar.button("Evaluate RAG System"):
    try:
        with st.spinner("Running evaluation..."):
            headers = {
                "X-API-Key": API_KEY
            }
            response = requests.post(f"{BACKEND_URL}/evaluate", headers=headers)
            response.raise_for_status()
            evaluation_results = response.json()

            st.subheader("Evaluation Results")
            st.metric(label="Average Precision", value=f"{evaluation_results['average_precision']:.2f}")
            st.metric(label="Average Recall", value=f"{evaluation_results['average_recall']:.2f}")

            st.markdown("### Detailed Results per Query")
            for detail in evaluation_results['detailed_results']:
                st.json(detail)

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the backend is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during evaluation: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during evaluation: {e}")
