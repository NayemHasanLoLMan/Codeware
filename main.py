import json
import os
import asyncio
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import chromadb
from chromadb.config import Settings
import logging
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multilingual RAG Chatbot", version="1.0.0")

# Pydantic models
class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    is_flow_triggered: bool = False
    trigger_id: Optional[str] = None

class FlowTriggerRequest(BaseModel):
    user_id: str
    trigger_id: str

# Global variables
chatbot_flow = {}
embedding_model = None
chroma_client = None
collection = None

class FlowMatcher:
    
    
    def __init__(self, flow_data: Dict):
        self.flow_data = flow_data
        self.keyword_mappings = self._build_keyword_mappings()
    
    def _build_keyword_mappings(self) -> Dict[str, str]:
        
        mappings = {}
        
        for flow_item in self.flow_data:
            if 'keywords' in flow_item:
                trigger_id = flow_item.get('id')
                for keyword in flow_item['keywords']:
                    mappings[keyword.lower()] = trigger_id
            
            # Also map option labels to their trigger IDs
            if 'options' in flow_item:
                for option in flow_item['options']:
                    if 'trigger' in option and 'label' in option:
                        mappings[option['label'].lower()] = option['trigger']
        
        return mappings
    
    def find_trigger(self, query: str) -> Optional[str]:
        
        query_lower = query.lower()
        
        # Direct keyword matching
        for keyword, trigger_id in self.keyword_mappings.items():
            if keyword in query_lower:
                logger.info(f"Found keyword match: {keyword} -> {trigger_id}")
                return trigger_id
        
        # Service-specific intent matching
        service_intents = {
            'packages': ['679e564098ea05fc9dd74968_ad3734fab0d51f1a', '679e564098ea05fc9dd74969_21f51a84a89a4f56'],
            'new connection': ['679e564098ea05fc9dd74964_5b703bf48a2b99f0', '679e564098ea05fc9dd74965_e9127afca7661432'],
            'bill pay': ['679e564098ea05fc9dd7496c_1a827ff9bcbc67a2', '679e564098ea05fc9dd7496d_2a872af5967812ec'],
            'service request': ['679e564098ea05fc9dd7497a_4ca15b5e495f38cd', '679e564098ea05fc9dd7497b_975e1f9485e97c15'],
            'coverage': ['679e564098ea05fc9dd7498a_9f13bae58a3a5d98', '679e564098ea05fc9dd7498b_6592e3a42d72d0fc'],
            'dial app': ['686e4e5b57956d829940ef60_12898e0dff1aa3a1', '686e4e9c57956d829940ef61_269b48970ec086e6']
        }
        
        # Bengali equivalents
        bengali_intents = {
            'প্যাকেজ': 'packages',
            'নতুন সংযোগ': 'new connection',
            'বিল পে': 'bill pay',
            'সার্ভিস রিকোয়েস্ট': 'service request',
            'কাভারেজ': 'coverage',
            'ডায়াল অ্যাপ': 'dial app'
        }
        
        # Check Bengali intents first
        for bengali_term, english_term in bengali_intents.items():
            if bengali_term in query:
                triggers = service_intents.get(english_term, [])
                if triggers:
                    return triggers[0]  # Return first trigger
        
        # Check English intents
        for intent, triggers in service_intents.items():
            if intent in query_lower:
                return triggers[0]  # Return first trigger
        
        return None

class OllamaLLM(LLM):
    
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                return "I'm experiencing technical difficulties. Please try again later."
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return "I'm currently unable to process your request. Please contact our customer service."

class RAGEngine:
    
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='paraphrase-multilingual-MiniLM-L12-v2',
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.llm = OllamaLLM()
        self.vectorstore = None
        
    def setup_vector_store(self):
        
        try:
            # Setup ChromaDB with LangChain
            persist_directory = "./data/chroma_langchain"
            
            # Initialize or load existing vector store
            if os.path.exists(persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing LangChain vector store")
                
                # Check if we need to reload documents
                if self.vectorstore._collection.count() == 0:
                    self._load_documents_from_json()
            else:
                # Create new vector store
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Created new LangChain vector store")
                self._load_documents_from_json()
                
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise
    
    def _load_sample_documents(self):
        
        sample_docs = [
            {
                "id": "doc1",
                "text": "iDesk360 offers high-speed internet packages ranging from 75 Mbps to 90 Mbps. Our premium package includes 90 Mbps speed with 500 minutes dial talktime for BDT 1900.",
                "metadata": {"source": "packages_info.txt"}
            },
            {
                "id": "doc2", 
                "text": "For new connections, customers can choose from multiple packages. Installation is free and typically completed within 3-5 business days.",
                "metadata": {"source": "connection_info.txt"}
            },
            {
                "id": "doc3",
                "text": "আমাদের সেবা সমূহ: ইন্টারনেট কানেকশন, বিল পেমেন্ট, প্যাকেজ আপগ্রেড, কাস্টমার সাপোর্ট। আমরা ২৪/৭ গ্রাহক সেবা প্রদান করি।",
                "metadata": {"source": "services_bangla.txt"}
            },
            {
                "id": "doc4",
                "text": "Bill payment can be done online, through mobile banking, or at our service centers. We accept bKash, Nagad, and Rocket payments.",
                "metadata": {"source": "payment_info.txt"}
            },
            {
                "id": "doc5",
                "text": "Coverage areas include Dhaka, Chittagong, Sylhet, and other major cities. Check coverage availability in your area before ordering.",
                "metadata": {"source": "coverage_info.txt"}
            }
        ]
        
        texts = [doc["text"] for doc in sample_docs]
        embeddings = self.embedding_model.encode(texts)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[doc["metadata"] for doc in sample_docs],
            ids=[doc["id"] for doc in sample_docs]
        )
        logger.info(f"Loaded {len(sample_docs)} sample documents")
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        
        try:
            query_embedding = self.embedding_model.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        
        try:
            # Prepare context from retrieved documents
            context = "\n".join([doc['content'] for doc in context_docs])
            
            # Create prompt for multilingual support
            prompt = f"""You are a helpful customer service assistant for iDesk360, a telecommunications company in Bangladesh. 
                        You can respond in English, Bengali/Bangla, or Banglish (Bengali written in English).

                        Context information:
                        {context}

                        User question: {query}

                        Please provide a helpful and accurate answer based on the context. If the context doesn't contain relevant information, politely say you don't have that specific information and suggest contacting customer service.

                        Answer:"""

            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2:3b",  # You can change this to your preferred model
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I'm experiencing technical difficulties. Please try again later."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama: {e}")
            return "I'm currently unable to process your request. Please contact our customer service."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "An error occurred while processing your request."

# Initialize components
rag_engine = RAGEngine()
flow_matcher = None

@app.on_event("startup")
async def startup_event():
    
    global flow_matcher, chatbot_flow
    
    try:
        # Load chatbot flow data
        if os.path.exists('codeware_bot_flow.json'):
            with open('codeware_bot_flow.json', 'r', encoding='utf-8') as f:
                chatbot_flow = json.load(f)
            flow_matcher = FlowMatcher(chatbot_flow)
            logger.info("Loaded chatbot flow data successfully")
        else:
            logger.error("codeware_bot_flow.json not found! Please ensure the file is in the project root.")
            raise FileNotFoundError("codeware_bot_flow.json is required for the application to run")
        
        # Setup vector store
        rag_engine.setup_vector_store()
        
        # Test Ollama connection
        try:
            test_response = requests.post(
                rag_engine.ollama_url,
                json={"model": "llama3.2:3b", "prompt": "Hello", "stream": False},
                timeout=10
            )
            if test_response.status_code == 200:
                logger.info("Ollama connection successful")
            else:
                logger.warning(f"Ollama connection test failed: {test_response.status_code}")
        except:
            logger.warning("Could not connect to Ollama. Make sure it's running on localhost:11434")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

async def call_external_chatbot_api(user_id: str, trigger_id: str) -> Dict:
    
    # In a real implementation, this would call your actual chatbot service
    return {
        "user_id": user_id,
        "trigger_id": trigger_id,
        "message": "Flow triggered successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that handles both RAG and rule-based flows"""
    try:
        # Step 1: Check for flow triggers
        trigger_id = flow_matcher.find_trigger(request.question)
        
        if trigger_id:
            # Trigger rule-based flow
            logger.info(f"Triggering flow {trigger_id} for user {request.user_id}")
            
            # Call external chatbot API
            flow_response = await call_external_chatbot_api(request.user_id, trigger_id)
            
            return ChatResponse(
                answer="I've connected you to our service flow. An agent will assist you shortly.",
                sources=[],
                is_flow_triggered=True,
                trigger_id=trigger_id
            )
        
        # Step 2: RAG Answer Generation
        logger.info(f"Generating RAG answer for user {request.user_id}")
        
        # Retrieve relevant documents using LangChain
        relevant_docs = rag_engine.retrieve_documents(request.question)
        
        if not relevant_docs:
            return ChatResponse(
                answer="I don't have specific information about your query. Please contact our customer service for assistance.",
                sources=[]
            )
        
        # Generate answer using LangChain + Ollama
        answer = rag_engine.generate_answer(request.question, relevant_docs)
        
        # Extract sources from LangChain documents
        sources = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
        
        return ChatResponse(
            answer=answer,
            sources=list(set(sources))  # Remove duplicates
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chatbot")
async def chatbot_flow_endpoint(request: FlowTriggerRequest):
    
    try:
        # Find the flow item by trigger_id
        flow_item = None
        for item in chatbot_flow:
            if item.get('id') == request.trigger_id:
                flow_item = item
                break
        
        if not flow_item:
            raise HTTPException(status_code=404, detail="Flow not found")
        
        # Return the flow message and options
        response = {
            "user_id": request.user_id,
            "trigger_id": request.trigger_id,
            "message": flow_item.get('message', ''),
            "options": flow_item.get('options', []),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chatbot flow endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_available": await check_ollama_health(),
        "vector_store_ready": rag_engine.vectorstore is not None
    }

async def check_ollama_health() -> bool:
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

@app.get("/flows")
async def get_flows():
    
    return {
        "flows": chatbot_flow,
        "total_flows": len(chatbot_flow)
    }

@app.get("/")
async def root():
    
    return {
        "message": "Multilingual RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Main chat endpoint",
            "chatbot": "POST /chatbot - Flow trigger endpoint", 
            "health": "GET /health - Health check",
            "flows": "GET /flows - Get available flows"
        },
        "supported_languages": ["English", "Bengali/Bangla", "Banglish"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
