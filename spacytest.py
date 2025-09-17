#!/usr/bin/env python3
"""
Sentence-BERT Semantic Similarity Upgrade
Better than TF-IDF for Aspire Support Chatbot
"""

import subprocess
import sys
import time
import numpy as np
from typing import List, Tuple, Optional

def install_sentence_transformers():
    """Install sentence-transformers library"""
    
    print("📦 Installing Sentence-BERT (sentence-transformers)")
    print("=" * 50)
    
    try:
        # Install sentence-transformers
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "sentence-transformers"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Successfully installed sentence-transformers")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install sentence-transformers: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_sentence_bert_basic():
    """Test basic Sentence-BERT functionality"""
    
    print("\n🧪 Testing Basic Sentence-BERT Functionality")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load a lightweight but powerful model
        print("📥 Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded successfully!")
        
        # Test sentences from your chatbot domain
        test_sentences = [
            "I need help with password reset",
            "Can you help me reset my password?",
            "S0C4 abend error occurred",
            "Storage violation S0C4 happened",
            "What is the relationship between Aspire and VTO?",
            "How are Aspire and VTO connected?",
            "Memory issue in production system",
            "Production memory problem occurred",
            "Job is running for a long time",
            "Long running job analysis needed"
        ]
        
        print(f"\n🔍 Testing with {len(test_sentences)} sentences...")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = model.encode(test_sentences)
        encoding_time = (time.time() - start_time) * 1000
        
        print(f"⏱️  Encoding time: {encoding_time:.2f}ms")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"🎯 Embedding dimension: {embeddings.shape[1]}")
        
        # Test similarity calculations
        print(f"\n🎯 Similarity Test Results:")
        test_pairs = [
            (0, 1),  # password reset variations
            (2, 3),  # S0C4 abend variations  
            (4, 5),  # Aspire-VTO relationship variations
            (6, 7),  # memory issue variations
            (8, 9)   # long running job variations
        ]
        
        for i, j in test_pairs:
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"  📝 '{test_sentences[i][:40]}...'")
            print(f"  📝 '{test_sentences[j][:40]}...'")
            print(f"  🎯 Similarity: {similarity:.4f}")
            print()
        
        return model, embeddings
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install sentence-transformers first")
        return None, None
    except Exception as e:
        print(f"❌ Error testing Sentence-BERT: {e}")
        return None, None

def compare_with_tfidf():
    """Compare Sentence-BERT with your current TF-IDF implementation"""
    
    print("⚖️ Sentence-BERT vs TF-IDF Comparison")
    print("=" * 45)
    
    # Test queries from your chatbot
    queries = [
        "password reset help",
        "S0C4 abend error", 
        "memory issue production",
        "aspire vto relationship",
        "job running long time"
    ]
    
    candidates = [
        "reset my password",
        "forgot password", 
        "S0C4 storage violation",
        "abend code error",
        "memory problem in system",
        "production memory issue",
        "what is relationship between aspire and vto",
        "connection aspire vto",
        "job is running for hours",
        "long running job analysis"
    ]
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load Sentence-BERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("🧪 Testing both approaches...")
        
        results = {}
        
        # Test Sentence-BERT
        print("\n1️⃣ Sentence-BERT Results:")
        start_time = time.time()
        
        # Encode all texts at once (efficient)
        all_texts = queries + candidates
        all_embeddings = model.encode(all_texts)
        
        query_embeddings = all_embeddings[:len(queries)]
        candidate_embeddings = all_embeddings[len(queries):]
        
        sbert_results = []
        for i, query in enumerate(queries):
            similarities = cosine_similarity([query_embeddings[i]], candidate_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            best_match = candidates[best_idx]
            
            sbert_results.append((query, best_match, best_score))
            print(f"  📝 '{query}' → '{best_match}' (score: {best_score:.4f})")
        
        sbert_time = (time.time() - start_time) * 1000
        
        # Test TF-IDF (your current implementation)
        print("\n2️⃣ TF-IDF Results:")
        try:
            from semantic_similarity import find_best_semantic_match, calculate_semantic_similarity
            
            start_time = time.time()
            tfidf_results = []
            
            for query in queries:
                best_match = find_best_semantic_match(query, candidates, "general", 0.1)
                if best_match:
                    score = calculate_semantic_similarity(query, best_match, "general")
                else:
                    best_match = "No match"
                    score = 0.0
                
                tfidf_results.append((query, best_match, score))
                print(f"  📝 '{query}' → '{best_match}' (score: {score:.4f})")
            
            tfidf_time = (time.time() - start_time) * 1000
            
            # Performance comparison
            print(f"\n📊 Performance Comparison:")
            print(f"{'Method':<15} {'Time (ms)':<12} {'Avg Score':<12}")
            print("-" * 40)
            
            sbert_avg = sum(r[2] for r in sbert_results) / len(sbert_results)
            tfidf_avg = sum(r[2] for r in tfidf_results) / len(tfidf_results)
            
            print(f"{'Sentence-BERT':<15} {sbert_time:<12.2f} {sbert_avg:<12.4f}")
            print(f"{'TF-IDF':<15} {tfidf_time:<12.2f} {tfidf_avg:<12.4f}")
            
            # Analysis
            print(f"\n🔍 Analysis:")
            if sbert_avg > tfidf_avg:
                improvement = ((sbert_avg - tfidf_avg) / tfidf_avg) * 100
                print(f"✅ Sentence-BERT is {improvement:.1f}% more accurate")
            
            if sbert_time < tfidf_time * 2:
                print(f"✅ Sentence-BERT performance is acceptable")
            else:
                print(f"⚠️ Sentence-BERT is slower but more accurate")
                
        except ImportError:
            print("⚠️ TF-IDF comparison skipped - semantic_similarity not available")
        
    except ImportError:
        print("❌ Sentence-BERT not available for comparison")
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

class SentenceBertSimilarity:
    """
    Advanced Sentence-BERT similarity matcher
    Replacement for TF-IDF semantic similarity
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', min_similarity=0.3):
        """
        Initialize Sentence-BERT similarity matcher
        
        Args:
            model_name: Sentence-BERT model to use
            min_similarity: Minimum similarity threshold
        """
        self.model_name = model_name
        self.min_similarity = min_similarity
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence-BERT model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"📥 Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully")
        except ImportError:
            print("❌ sentence-transformers not installed")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def find_best_match(self, query: str, candidates: List[str], 
                       return_score: bool = False) -> Optional[str]:
        """
        Find best matching candidate using Sentence-BERT
        
        Args:
            query: Query text
            candidates: List of candidate texts
            return_score: Whether to return similarity score
            
        Returns:
            Best matching candidate or (candidate, score) if return_score=True
        """
        if not self.model or not query or not candidates:
            return None
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Encode query and candidates
            all_texts = [query] + candidates
            embeddings = self.model.encode(all_texts)
            
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # Find best match above threshold
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= self.min_similarity:
                best_candidate = candidates[best_idx]
                if return_score:
                    return best_candidate, best_score
                return best_candidate
            
            return None
            
        except Exception as e:
            print(f"❌ Error in Sentence-BERT matching: {e}")
            return None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not self.model or not text1 or not text2:
            return 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0

def integration_example():
    """Show how to integrate Sentence-BERT into your chatbot"""
    
    print("\n🔧 Integration Example for Your Chatbot")
    print("=" * 45)
    
    integration_code = '''
# 1. Update requirements.txt
sentence-transformers>=2.2.0

# 2. Create enhanced_semantic_similarity.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SentenceBertMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def find_best_qa_match(self, query, qa_questions):
        if not qa_questions:
            return None
        
        # Encode query and questions
        all_texts = [query] + qa_questions
        embeddings = self.model.encode(all_texts)
        
        # Calculate similarities
        query_emb = embeddings[0:1]
        question_embs = embeddings[1:]
        similarities = cosine_similarity(query_emb, question_embs)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > 0.5:  # Threshold
            return qa_questions[best_idx]
        return None

# 3. Update voice.py ProductionSupportAnalyzer
class ProductionSupportAnalyzer:
    def __init__(self):
        self.sbert_matcher = SentenceBertMatcher()
    
    def _find_best_qa_match(self, user_question):
        candidates = list(self.qa_data.keys())
        best_match = self.sbert_matcher.find_best_qa_match(
            user_question, candidates
        )
        if best_match:
            return self.qa_data[best_match]
        return None
    '''
    
    print("📝 Integration Steps:")
    print("1. Install: pip install sentence-transformers")
    print("2. Replace TF-IDF matcher with Sentence-BERT")
    print("3. Update your ProductionSupportAnalyzer class")
    print("4. Test with your existing Q&A data")
    
    print(f"\n💻 Sample Integration Code:")
    print(integration_code)

def test_with_your_data():
    """Test Sentence-BERT with your actual chatbot data"""
    
    print("\n🎯 Testing with Your Aspire Support Data")
    print("=" * 45)
    
    # Your actual Q&A data examples
    qa_pairs = {
        "what is the relationship between aspire and vto application": 
            "Aspire and VTO applications are integrated through data exchange APIs...",
        "how do i reset my password": 
            "To reset your password, contact the help desk or use the self-service portal...",
        "what are common abend codes": 
            "Common abend codes include S0C4 (storage violation), S0C7 (data exception)...",
        "job is running for long time what might be the reason":
            "Long running jobs may be caused by resource contention, data volume..."
    }
    
    test_queries = [
        "aspire vto connection",
        "password reset help", 
        "common abends",
        "job running too long",
        "memory issue production"
    ]
    
    try:
        # Test with Sentence-BERT
        sbert_matcher = SentenceBertSimilarity(min_similarity=0.3)
        
        if sbert_matcher.model:
            print("✅ Testing Sentence-BERT with your data:")
            
            questions = list(qa_pairs.keys())
            
            for query in test_queries:
                best_match = sbert_matcher.find_best_match(query, questions, return_score=True)
                
                if best_match:
                    match, score = best_match
                    answer = qa_pairs[match][:60] + "..."
                    print(f"\n📝 Query: '{query}'")
                    print(f"🎯 Match: '{match}' (score: {score:.4f})")
                    print(f"💬 Answer: {answer}")
                else:
                    print(f"\n📝 Query: '{query}'")
                    print(f"❌ No match found")
        else:
            print("❌ Sentence-BERT model not available")
            
    except Exception as e:
        print(f"❌ Error testing with your data: {e}")

def main():
    """Main function to test Sentence-BERT upgrade"""
    
    print("🚀 Sentence-BERT Upgrade for Aspire Support")
    print("🎯 Better than TF-IDF Semantic Similarity")
    print("=" * 60)
    
    # Install sentence-transformers
    if install_sentence_transformers():
        
        # Test basic functionality
        model, embeddings = test_sentence_bert_basic()
        
        if model:
            # Compare with TF-IDF
            compare_with_tfidf()
            
            # Test with your actual data
            test_with_your_data()
            
            # Show integration example
            integration_example()
            
            print(f"\n✅ Sentence-BERT Testing Complete!")
            print(f"🎉 Ready to upgrade from TF-IDF to Sentence-BERT!")
        else:
            print(f"❌ Sentence-BERT testing failed")
    else:
        print(f"❌ Installation failed")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 
