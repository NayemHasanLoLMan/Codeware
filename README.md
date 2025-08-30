# Multilingual RAG Chatbot with Rule-based Flow

A Q&A chatbot API that combines RAG (Retrieval-Augmented Generation) pipeline with rule-based flow triggers, supporting multiple languages (English, Bengali, and Banglish). The system uses local Ollama LLM for privacy and cost-effectiveness.

## Features

- **Multilingual Support**: English, Bengali/Bangla, and Banglish
- **RAG Pipeline**: LangChain + ChromaDB Vector Database + Local Ollama LLM
- **Rule-based Flows**: JSON-defined service flows with keyword matching
- **Local LLM**: Uses Ollama for privacy and no external API dependencies
- **FastAPI Backend**: Modern, fast, and well-documented API
- **Docker Support**: Easy deployment with Docker and docker-compose

## Architecture

The system uses your provided `codeware_bot_flow.json` file for two purposes:

1. **Rule-based Flow Triggers**: Matches user queries to predefined service flows
2. **RAG Document Source**: Extracts all text content (messages, options, carousel titles) as documents for semantic search

```
User Query → Flow Trigger Check → RAG Pipeline → Response
                     ↓                 ↓
              External API Call    Vector Search on JSON Content + Local LLM
```

### Flow Logic:
1. **Step 1**: Check if query matches keywords from `codeware_bot_flow.json`
2. **Step 2**: If match found, trigger external chatbot API with appropriate trigger_id
3. **Step 3**: If no match, perform semantic search on all JSON content and generate answer using local Ollama LLM

## Data Source

The application uses your `codeware_bot_flow.json` file as the primary data source:
- **Flow Triggers**: Keywords and service intents from the JSON
- **RAG Documents**: All text content (messages, options, carousel titles, keywords) extracted from the JSON
- **No external documents needed**: The 3000+ line JSON file provides comprehensive service information

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Ollama (for local LLM)

## Setup Instructions

### Option 1: Docker Setup (Recommended)

1. **Clone and Setup**:
```bash
git clone https://github.com/NayemHasanLoLMan/Codeware.git
cd rag-chatbot
```

2. **Copy your JSON flow file**:
```bash
# Copy your codeware_bot_flow.json to the project root (REQUIRED)
cp /path/to/your/codeware_bot_flow.json .
```

3. **Run setup validation** (optional but recommended):
```bash
python setup.py
```

4. **Start with Docker Compose**:
```bash
docker-compose up -d
```

5. **Download Ollama Model**:
```bash
# Wait for Ollama to start, then download the model
docker exec ollama ollama pull llama3.2:3b
```

6. **Verify Setup**:
```bash
curl http://localhost:8000/health
```

### Option 2: Local Development Setup

1. **Install Ollama**:
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

2. **Download Model**:
```bash
ollama pull llama3.2:3b
```

3. **Copy your JSON file and setup**:
```bash
# Copy your codeware_bot_flow.json to the project root (REQUIRED)
cp /path/to/your/codeware_bot_flow.json .

# Install dependencies
pip install -r requirements.txt

# Run setup validation
python setup.py
```

4. **Run the Application**:
```bash
python main.py
```

## Analyzing Your JSON File

Before running the application, you can analyze your `codeware_bot_flow.json`:

```bash
python data_loader.py
```

This will show:
- Total number of flow items
- Languages detected
- Sample messages and keywords
- Structure validation
- Documents extracted for RAG

## API Endpoints

### Main Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "user_id": "12345",
  "question": "I want to know about packages"
}
```

**Response**:
```json
{
  "answer": "Our packages include Basic, Premium, and Business plans...",
  "sources": ["packages_info.txt"],
  "is_flow_triggered": false,
  "trigger_id": null
}
```

### Flow Trigger Endpoint
```http
POST /chatbot
Content-Type: application/json

{
  "user_id": "12345",
  "trigger_id": "679e564098ea05fc9dd74968_ad3734fab0d51f1a"
}
```

### Health Check
```http
GET /health
```

### Available Flows
```http
GET /flows
```

## Example Usage

### RAG Query (General Question)
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "question": "What are your internet speeds?"
     }'
```

### Flow Trigger (Service Request)
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123", 
       "question": "I want to see packages"
     }'
```

### Bengali Query
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "question": "আমার ইন্টারনেট প্যাকেজ সম্পর্কে জানতে চাই"
     }'
```

## Project Structure

```
rag-chatbot/
├── main.py                    # Main FastAPI application
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Multi-service Docker setup
├── codeware_bot_flow.json    # Chatbot flow definitions
├── README.md                 # This file
└── data/                     # Data directory (created automatically)
```

## Configuration

### Environment Variables

- `OLLAMA_URL`: Ollama API URL (default: http://localhost:11434)
- `LOG_LEVEL`: Logging level (default: INFO)

### Supported Ollama Models

- `llama3.2:3b` (default, recommended for speed)
- `llama3.2:1b` (smaller, faster)
- `mistral:7b` (alternative option)

Change the model in `main.py`:
```python
"model": "your-preferred-model"
```

## Language Support

The chatbot supports:

- **English**: "What are your packages?"
- **Bengali**: "আপনাদের প্যাকেজ কি কি আছে?"
- **Banglish**: "Apnader package gulo ki ki ache?"

### Trigger Keywords

The system recognizes service-related keywords in all supported languages:

- **Packages**: packages, প্যাকেজ, package, plan
- **New Connection**: new connection, নতুন সংযোগ
- **Bill Pay**: bill pay, বিল পে, payment
- **Service Request**: service request, সার্ভিস রিকোয়েস্ট
- **Coverage**: coverage, কাভারেজ

## Performance Optimization

- Vector embeddings cached in ChromaDB
- Efficient similarity search with sentence-transformers
- Local LLM eliminates API latency
- Docker multi-stage builds for smaller images

## Monitoring and Debugging

### Check Logs
```bash
# Docker
docker-compose logs -f chatbot-api
docker-compose logs -f ollama

# Local
python main.py  # See logs in terminal
```

### Test Ollama Connection
```bash
curl http://localhost:11434/api/tags
```

### Test Vector Store
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

1. **Ollama not responding**:
   - Ensure Ollama is running: `ollama serve`
   - Check if model is downloaded: `ollama list`
   - Verify port 11434 is available

2. **Vector store errors**:
   - Clear ChromaDB data: `rm -rf ./data/chroma`
   - Restart the application

3. **Memory issues**:
   - Use smaller Ollama model: `llama3.2:3b`
   - Increase Docker memory allocation

4. **Slow responses**:
   - Use GPU acceleration with Ollama
   - Reduce context window size
   - Use lighter embedding model

## Scaling Considerations

- **Production**: Use GPU-enabled instances for Ollama
- **Load Balancing**: Multiple API instances behind load balancer
- **Database**: Replace ChromaDB with production vector database (Pinecone, Weaviate)
- **Caching**: Implement Redis for frequently asked questions

## Testing

Test the API with different scenarios:

```python
import requests

# Test RAG
response = requests.post("http://localhost:8000/chat", json={
    "user_id": "test123",
    "question": "What internet speeds do you offer?"
})

# Test Flow Trigger
response = requests.post("http://localhost:8000/chat", json={
    "user_id": "test123", 
    "question": "I want to see packages"
})

