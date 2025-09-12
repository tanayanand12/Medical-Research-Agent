# Use an official Python runtime as a parent image
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the files
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r working-requirements.txt

# call download_pubmed_embeddings.py to download the PubMed embeddings
RUN python download_pubmed_embeddings.py

# Run the FastAPI application using Uvicorn
CMD ["python", "research_agent_api.py"]
