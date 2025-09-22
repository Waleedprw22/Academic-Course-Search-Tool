# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the container
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 is necessary to make the server accessible from outside the container
# --timeout 300 (or higher) is important for evaluation endpoint
# Environment variables for API keys will be passed at runtime (e.g., docker run -e)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout", "300"]
