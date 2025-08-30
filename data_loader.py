#!/usr/bin/env python3

import json
import re
from typing import Dict, List, Any
from collections import Counter

def load_and_analyze_flow_data(file_path: str = 'codeware_bot_flow.json') -> Dict[str, Any]:
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Successfully loaded {file_path}")
        print(f"Total flow items: {len(data)}")
        
        # Analyze the structure
        analysis = {
            'total_items': len(data),
            'items_with_messages': 0,
            'items_with_options': 0,
            'items_with_keywords': 0,
            'items_with_carousels': 0,
            'total_keywords': 0,
            'total_options': 0,
            'languages_detected': set(),
            'message_samples': [],
            'keyword_samples': []
        }
        
        for item in data:
            # Count different types of content
            if 'message' in item and item['message']:
                analysis['items_with_messages'] += 1
                
                # Sample some messages
                if len(analysis['message_samples']) < 5:
                    analysis['message_samples'].append(item['message'][:100] + "..." if len(item['message']) > 100 else item['message'])
                
                # Detect languages
                if contains_bengali(item['message']):
                    analysis['languages_detected'].add('Bengali')
                if contains_english(item['message']):
                    analysis['languages_detected'].add('English')
            
            if 'options' in item and item['options']:
                analysis['items_with_options'] += 1
                analysis['total_options'] += len(item['options'])
            
            if 'keywords' in item and item['keywords']:
                analysis['items_with_keywords'] += 1
                analysis['total_keywords'] += len(item['keywords'])
                
                # Sample some keywords
                if len(analysis['keyword_samples']) < 10:
                    analysis['keyword_samples'].extend(item['keywords'][:3])
            
            if 'carousel' in item and item['carousel']:
                analysis['items_with_carousels'] += 1
        
        # Convert set to list for JSON serialization
        analysis['languages_detected'] = list(analysis['languages_detected'])
        
        return data, analysis
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        print("Please ensure the codeware_bot_flow.json file is in the project directory.")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}")
        print(f"Details: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {e}")
        return None, None

def contains_bengali(text: str) -> bool:
    
    bengali_pattern = re.compile(r'[\u0980-\u09FF]')
    return bool(bengali_pattern.search(text))

def contains_english(text: str) -> bool:
    
    english_pattern = re.compile(r'[a-zA-Z]')
    return bool(english_pattern.search(text))

def extract_rag_documents(flow_data: List[Dict]) -> List[Dict]:
    
    documents = []
    
    for i, flow_item in enumerate(flow_data):
        flow_id = flow_item.get('id', f'flow_{i}')
        
        # Extract message content
        if 'message' in flow_item and flow_item['message']:
            # Clean and prepare message text
            message_text = clean_text(flow_item['message'])
            if message_text.strip():
                documents.append({
                    "id": f"msg_{flow_id}",
                    "text": message_text,
                    "metadata": {
                        "source": "codeware_bot_flow.json",
                        "type": "message",
                        "flow_id": flow_id,
                        "language": detect_language(message_text)
                    }
                })
        
        # Extract option labels as searchable content
        if 'options' in flow_item and flow_item['options']:
            option_texts = []
            for option in flow_item['options']:
                if 'label' in option and option['label']:
                    option_texts.append(option['label'])
            
            if option_texts:
                combined_options = "Available options: " + ", ".join(option_texts)
                documents.append({
                    "id": f"opts_{flow_id}",
                    "text": combined_options,
                    "metadata": {
                        "source": "codeware_bot_flow.json",
                        "type": "options",
                        "flow_id": flow_id,
                        "language": detect_language(combined_options)
                    }
                })
        
        # Extract carousel content
        if 'carousel' in flow_item and flow_item['carousel']:
            for j, carousel_item in enumerate(flow_item['carousel']):
                if 'title' in carousel_item and carousel_item['title']:
                    # Clean HTML tags and extract meaningful content
                    clean_title = clean_text(carousel_item['title'])
                    if clean_title.strip():
                        documents.append({
                            "id": f"car_{flow_id}_{j}",
                            "text": f"Service package: {clean_title}",
                            "metadata": {
                                "source": "codeware_bot_flow.json",
                                "type": "carousel",
                                "flow_id": flow_id,
                                "carousel_index": j,
                                "language": detect_language(clean_title)
                            }
                        })
        
        # Extract keywords as context
        if 'keywords' in flow_item and flow_item['keywords']:
            keywords_text = "Related search terms: " + ", ".join(flow_item['keywords'])
            documents.append({
                "id": f"kw_{flow_id}",
                "text": keywords_text,
                "metadata": {
                    "source": "codeware_bot_flow.json",
                    "type": "keywords",
                    "flow_id": flow_id,
                    "language": "mixed"
                }
            })
    
    print(f"Extracted {len(documents)} documents for RAG indexing")
    return documents

def clean_text(text: str) -> str:
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text.strip()

def detect_language(text: str) -> str:
    
    if contains_bengali(text):
        if contains_english(text):
            return "mixed"
        return "bengali"
    elif contains_english(text):
        return "english"
    else:
        return "unknown"

def print_analysis(analysis: Dict[str, Any]):

    print("\n" + "="*50)
    print("FLOW DATA ANALYSIS")
    print("="*50)
    
    print(f"Total Items: {analysis['total_items']}")
    print(f"Items with Messages: {analysis['items_with_messages']}")
    print(f"Items with Options: {analysis['items_with_options']}")
    print(f"Items with Keywords: {analysis['items_with_keywords']}")
    print(f"Items with Carousels: {analysis['items_with_carousels']}")
    print(f"Total Keywords: {analysis['total_keywords']}")
    print(f"Total Options: {analysis['total_options']}")
    print(f"Languages Detected: {', '.join(analysis['languages_detected'])}")
    
    print("\nSample Messages:")
    for i, msg in enumerate(analysis['message_samples'][:3], 1):
        print(f"  {i}. {msg}")
    
    print("\nSample Keywords:")
    print(f"  {', '.join(analysis['keyword_samples'][:10])}")

def validate_flow_structure(flow_data: List[Dict]) -> bool:
    """Validate the structure of flow data"""
    required_fields = ['id']
    issues = []
    
    for i, item in enumerate(flow_data):
        for field in required_fields:
            if field not in item:
                issues.append(f"Item {i}: Missing required field '{field}'")
        
        # Check for circular references in triggers
        if 'options' in item:
            for option in item['options']:
                if 'trigger' in option and option['trigger'] == item.get('id'):
                    issues.append(f"Item {i}: Circular reference detected")
    
    if issues:
        print("\nVALIDATION ISSUES:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        return False
    
    print("\nFlow structure validation passed")
    return True

if __name__ == "__main__":
    print("Analyzing codeware_bot_flow.json...")
    
    # Load and analyze the data
    flow_data, analysis = load_and_analyze_flow_data()
    
    if flow_data and analysis:
        print_analysis(analysis)
        
        # Validate structure
        validate_flow_structure(flow_data)
        
        # Extract documents for RAG
        rag_docs = extract_rag_documents(flow_data)
        
        print(f"\nReady for RAG indexing with {len(rag_docs)} documents")
        print("\nRun the main application with: python main.py")
    else:
        print("\nFailed to load flow data. Please check the file and try again.")
