# Medical Research RAG System

A Retrieval-Augmented Generation (RAG) system for processing medical research papers and generating evidence-based answers using OpenAI's GPT-4 and FAISS vector database.

## Features

- PDF processing with intelligent chunking and metadata extraction
- Vector embeddings using OpenAI's text-embedding-ada-002
- Efficient similarity search using FAISS vector database
- RAG-based answer generation with GPT-4
- RESTful API with FastAPI
- Comprehensive error handling and logging
- Document citation tracking
- Index management system

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Windows/Linux/MacOS
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-research-rag.git
cd medical-research-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
medical-research-rag/
├── api_module.py         # FastAPI server implementation
├── pdf_processor.py      # PDF processing and chunking
├── vectorization.py      # Document embedding using OpenAI
├── faiss_db_manager.py   # Vector database management
├── rag_module.py         # RAG implementation with GPT-4
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## API Endpoints

### 1. Upload PDFs
```http
POST /upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]
index_name: string (optional, default="default")
```

### 2. Query Index
```http
POST /query
Content-Type: application/json

{
    "query": "What are the main findings about treatment efficacy?",
    "index_name": "medical_papers",
    "top_k": 5
}
```

### 3. List Indexes
```http
GET /indexes
```

### 4. Delete Index
```http
DELETE /indexes/{index_name}
```

## Usage Examples

1. Start the API server:
```bash
python api_module.py
```

2. Access the Swagger documentation:
```
http://localhost:8000/docs
```

3. Using curl:
```bash
# Upload PDFs
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  -F "index_name=medical_papers"

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about treatment efficacy?",
    "index_name": "medical_papers",
    "top_k": 5
  }'
```

4. Using Python requests:
```python
import requests

# Upload PDFs
files = [
    ('files', open('paper1.pdf', 'rb')),
    ('files', open('paper2.pdf', 'rb'))
]
response = requests.post(
    "http://localhost:8000/upload",
    files=files,
    data={"index_name": "medical_papers"}
)

# Query the system
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the main findings?",
        "index_name": "medical_papers",
        "top_k": 5
    }
)
print(response.json())
```

## Configuration

Key parameters can be adjusted in their respective modules:

- `PDFProcessor`: chunk size and overlap percentage
- `VectorizationModule`: embedding model and batch size
- `RAGModule`: GPT model, temperature, and max tokens
- `FaissVectorDB`: vector dimension and similarity metrics

## Error Handling

The system implements comprehensive error handling:
- Input validation
- PDF processing errors
- API rate limiting
- Database operations
- File system operations
- OpenAI API errors

## Logging

Logs are available at different levels:
- API endpoint access and errors
- PDF processing status
- Embedding operations
- Database operations
- RAG system responses

## Performance Considerations

- Use batch processing for large PDF collections
- Implement rate limiting for OpenAI API calls
- Monitor vector database size and performance
- Consider chunking parameters for optimal retrieval
- Cache frequently accessed embeddings

## Security Notes

- API key protection
- Input sanitization
- CORS configuration
- File upload restrictions
- Access control implementation

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please open an issue in the GitHub repository or contact [your-email].