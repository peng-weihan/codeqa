#!/usr/bin/env python3
"""
Test script for Voyage AI embedding integration
"""


import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append("/data3/pwh/codeqa")

from repo_qa_generator.rag.func_chunk_rag import VoyageEmbeddingModel

def test_voyage_embedding():
    """Test the Voyage AI embedding model"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not found in environment variables")
        print("Please set your Voyage AI API key in the .env file")
        return False
    
    try:
        # Initialize the embedding model
        print("🔧 Initializing Voyage AI embedding model...")
        embed_model = VoyageEmbeddingModel(api_key)
        print("✅ Voyage AI embedding model initialized successfully")
        
        # Test single text encoding
        print("\n🧪 Testing single text encoding...")
        test_text = "def hello_world(): print('Hello, World!')"
        embedding = embed_model.encode(test_text)
        print(f"✅ Single text encoding successful, embedding shape: {embedding.shape}")
        
        # Test batch encoding
        print("\n🧪 Testing batch encoding...")
        test_texts = [
            "def add(a, b): return a + b",
            "class Calculator: pass",
            "import numpy as np"
        ]
        embeddings = embed_model.encode(test_texts, batch_size=2)
        print(f"✅ Batch encoding successful, embeddings shape: {embeddings.shape}")
        
        # Test similarity calculation
        print("\n🧪 Testing similarity calculation...")
        query = "function to add numbers"
        query_embedding = embed_model.encode(query)[0]
        
        # Calculate similarities
        similarities = []
        for i, text in enumerate(test_texts):
            text_embedding = embeddings[i]
            similarity = np.dot(text_embedding, query_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(query_embedding)
            )
            similarities.append((text, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print("📊 Similarity rankings:")
        for i, (text, sim) in enumerate(similarities):
            print(f"  {i+1}. {text[:50]}... (similarity: {sim:.4f})")
        
        print("\n✅ All tests passed! Voyage AI embedding integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    import numpy as np
    success = test_voyage_embedding()
    sys.exit(0 if success else 1)
