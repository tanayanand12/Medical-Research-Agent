# FDA Agent

A comprehensive library for querying and analyzing FDA and clinical trial data using RAG (Retrieval-Augmented Generation).

## Installation

### As a Package
```bash
pip install git+https://github.com/yourusername/fda-agent.git
```

### From Source
```bash
git clone https://github.com/yourusername/fda-agent.git
cd fda-agent
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from fda_agent import FdaRAG

# Initialize with your OpenAI API key
rag = FdaRAG(api_key="your-openai-api-key")

# Make a query
result = rag.query("What are the recent adverse events for aspirin?")
print(result.answer)
print(f"Found {len(result.citations)} relevant citations")
```

### Quick Query
```python
from fda_agent import quick_query

answer = quick_query("What are the recent recalls for acetaminophen?")
print(answer)
```

### Advanced Configuration
```python
rag = FdaRAG(
    api_key="your-openai-api-key",
    model_name="gpt-4-turbo",
    embedding_model="text-embedding-ada-002",
    max_records=300,
    chunk_size=10000,
    chunk_overlap=400,
    max_context_length=8000
)
```

## Environment Variables

The following environment variables can be set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LOG_LEVEL`: Logging level (default: INFO)

## Features

- Query FDA drug labels, adverse events, and recalls
- RAG-based analysis with GPT-4
- Configurable chunking and context management
- Structured response format with citations
- Easy integration with other projects

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
