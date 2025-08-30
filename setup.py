#!/usr/bin/env python3


import os
import sys
import subprocess
import json
import requests
import time
from pathlib import Path

def check_file_exists(filename: str) -> bool:
    
    return Path(filename).exists()

def check_ollama_installation():
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Ollama is installed")
            return True
        else:
            print("Ollama is not installed")
            return False
    except FileNotFoundError:
        print("Ollama is not installed")
        return False

def check_ollama_service():
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama service is running")
            return True
        else:
            print("Ollama service is not responding")
            return False
    except requests.exceptions.RequestException:
        print("Ollama service is not running")
        return False

def install_ollama_model(model_name: str = "llama3.2:3b"):

    print(f"Installing Ollama model: {model_name}")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Model {model_name} installed successfully")
            return True
        else:
            print(f"Failed to install model {model_name}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error installing model: {e}")
        return False

def check_python_dependencies():
    
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import chromadb
        import requests
        print("All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing Python dependency: {e}")
        return False

def validate_json_file():
    
    if not check_file_exists('codeware_bot_flow.json'):
        print("codeware_bot_flow.json not found in project root")
        print("Please copy your codeware_bot_flow.json file to the project directory")
        return False
    
    try:
        with open('codeware_bot_flow.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("codeware_bot_flow.json should contain a list of flow items")
            return False
        
        print(f"codeware_bot_flow.json is valid with {len(data)} flow items")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return False

def start_ollama_service():
    
    print("Starting Ollama service...")
    try:
        # Try to start Ollama in background
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        for i in range(30):  # Wait up to 30 seconds
            if check_ollama_service():
                return True
            time.sleep(5)
            print(f"Waiting for Ollama to start... ({i+1}/30)")
        
        print("Ollama service failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False

def run_setup():
    
    print("RAG Chatbot Setup")
    
    # Check 1: JSON file
    print("\n1. Checking codeware_bot_flow.json...")
    if not validate_json_file():
        return False
    
    # Check 2: Python dependencies
    print("\n2. Checking Python dependencies...")
    if not check_python_dependencies():
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check 3: Ollama installation
    print("\n3. Checking Ollama installation...")
    if not check_ollama_installation():
        print("Please install Ollama from: https://ollama.ai/download")
        return False
    
    # Check 4: Ollama service
    print("\n4. Checking Ollama service...")
    if not check_ollama_service():
        if input("Start Ollama service? (y/n): ").lower() == 'y':
            if not start_ollama_service():
                return False
        else:
            print("Please start Ollama manually: ollama serve")
            return False
    
    # Check 5: Install model
    print("\n5. Checking Ollama model...")
    try:
        response = requests.post("http://localhost:11434/api/generate", 
                               json={"model": "llama3.2:3b", "prompt": "test", "stream": False},
                               timeout=10)
        if response.status_code == 404:
            if input("Install llama3.2:8b model? (y/n): ").lower() == 'y':
                if not install_ollama_model():
                    return False
            else:
                print("Model required for operation")
                return False
        else:
            print("Ollama model is available")
    except:
        if input("Install llama3.2:8b model? (y/n): ").lower() == 'y':
            if not install_ollama_model():
                return False
    
    print("\nSetup completed successfully!")
    print("\nTo start the application:")
    print("  Local: python main.py")
    print("  Docker: docker-compose up -d")
    print("\nAPI will be available at: http://localhost:8000")
    
    return True

def quick_test():
    
    print("\nRunning quick test...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("API health check passed")
        else:
            print("API health check failed")
            return False
        
        # Test chat endpoint
        test_query = {
            "user_id": "test_user",
            "question": "Hello, what services do you offer?"
        }
        
        response = requests.post("http://localhost:8000/chat", json=test_query, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("Chat endpoint test passed")
            print(f"Response: {result['answer'][:100]}...")
        else:
            print("Chat endpoint test failed")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"API test failed: {e}")
        print("Make sure the API is running: python main.py")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        run_setup()
