#!/usr/bin/env python3

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check():
    
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("Health check passed")
            print(f"   Ollama available: {health_data.get('ollama_available', False)}")
            print(f"   Vector store ready: {health_data.get('vector_store_ready', False)}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_chat_endpoint(query: str, expected_flow_trigger: bool = False) -> Dict[str, Any]:
    
    print(f"\nTesting query: '{query}'")
    
    try:
        payload = {
            "user_id": "test_user_123",
            "question": query
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("Chat request successful")
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Sources: {result.get('sources', [])}")
            print(f"   Flow triggered: {result.get('is_flow_triggered', False)}")
            
            if result.get('is_flow_triggered'):
                print(f"   Trigger ID: {result.get('trigger_id')}")
            
            return result
        else:
            print(f"Chat request failed: {response.status_code}")
            print(response.text)
            return {}
            
    except Exception as e:
        print(f"Chat request error: {e}")
        return {}

def test_flow_trigger_endpoint(trigger_id: str):
    """Test the chatbot flow endpoint"""
    print(f"\nTesting flow trigger: {trigger_id}")
    
    try:
        payload = {
            "user_id": "test_user_123",
            "trigger_id": trigger_id
        }
        
        response = requests.post(f"{BASE_URL}/chatbot", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("Flow trigger successful")
            print(f"   Message: {result.get('message', '')[:100]}...")
            print(f"   Options: {len(result.get('options', []))} available")
            return result
        else:
            print(f"Flow trigger failed: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"Flow trigger error: {e}")
        return {}

def test_multilingual_support():
    
    print("\nTesting multilingual support...")
    
    test_queries = [
        ("English", "What packages do you offer?"),
        ("Bengali", "আপনাদের প্যাকেজ গুলো কি কি?"),
        ("Banglish", "Apnader internet package gulo ki ache?"),
        ("Service Trigger", "I want to see packages"),
        ("Bengali Service", "প্যাকেজ দেখতে চাই")
    ]
    
    for lang, query in test_queries:
        print(f"\n--- {lang} ---")
        result = test_chat_endpoint(query)
        time.sleep(5)  # Small delay between requests

def run_comprehensive_tests():
    
    print("RAG CHATBOT API TESTS")
    
    # Test 1: Health check
    if not test_health_check():
        print("\nHealth check failed. Ensure the API is running.")
        return False
    
    # Test 2: Basic chat functionality
    print(f"\nTesting basic chat functionality...")
    basic_queries = [
        "Hello",
        "What services do you provide?",
        "How much does installation cost?",
        "Do you have coverage in Dhaka?"
    ]
    
    for query in basic_queries:
        test_chat_endpoint(query)
        time.sleep(1)
    
    # Test 3: Flow triggers
    print(f"\nTesting flow triggers...")
    trigger_queries = [
        "packages",
        "new connection", 
        "bill pay",
        "service request"
    ]
    
    for query in trigger_queries:
        result = test_chat_endpoint(query, expected_flow_trigger=True)
        
        # If flow was triggered, test the chatbot endpoint
        if result.get('is_flow_triggered') and result.get('trigger_id'):
            test_flow_trigger_endpoint(result['trigger_id'])
        
        time.sleep(5)
    
    # Test 4: Multilingual support
    test_multilingual_support()
    
    # Test 5: Error handling
    print(f"\nTesting error handling...")
    error_test_queries = [
        "",  # Empty query
        "x" * 1000,  # Very long query
        "completely unrelated query about quantum physics"  # Irrelevant query
    ]
    
    for query in error_test_queries:
        if query:
            test_chat_endpoint(query)
            time.sleep(1)
    
    print("\nAll tests completed!")

def performance_test():
    
    print("\nRunning performance test...")
    
    queries = [
        "What are your internet packages?",
        "I need new connection",
        "How to pay bill?",
        "আপনাদের সার্ভিস কি কি?",
        "Do you have coverage in my area?"
    ]
    
    total_time = 0
    successful_requests = 0
    
    for i, query in enumerate(queries * 2):  # Run each query twice
        start_time = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}/chat", 
                                   json={"user_id": f"perf_test_{i}", "question": query},
                                   timeout=20)
            
            end_time = time.time()
            request_time = end_time - start_time
            
            if response.status_code == 200:
                successful_requests += 1
                total_time += request_time
                print(f"Query {i+1}: {request_time:.2f}s")
            else:
                print(f"Query {i+1}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"Query {i+1}: Error - {e}")
        
        time.sleep(0.5)  # Small delay between requests
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        print(f"\nPerformance Results:")
        print(f"   Successful requests: {successful_requests}/{len(queries)*2}")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Total test time: {total_time:.2f}s")
    else:
        print("No successful requests in performance test")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "performance":
            performance_test()
        elif sys.argv[1] == "health":
            test_health_check()
        elif sys.argv[1] == "multilingual":
            test_multilingual_support()
        else:
            print("Usage: python test_api.py [performance|health|multilingual]")
    else:
        run_comprehensive_tests()