#!/usr/bin/env python3
"""
spaCy Model Performance vs Accuracy Comparison
Demonstrates the trade-offs between model size, speed, and accuracy
"""

import spacy
import time
import statistics
from typing import List, Dict, Tuple

# Sample texts for testing
TEST_TEXTS = [
    "Apple Inc. is planning to open a new store in New York City next month.",
    "The S0C4 abend occurred at 14:30 EST when processing the COBOL program.",
    "Microsoft Corporation reported quarterly earnings of $2.1 billion yesterday.",
    "The database connection failed with error code ORA-12154 in production.",
    "Google's headquarters in Mountain View, California employs over 50,000 people.",
    "System restart required after memory leak in application server JVM.",
    "Amazon Web Services launched three new data centers in Europe last week.",
    "Network timeout error 408 detected on mainframe system PROD01.",
    "Tesla's stock price increased by 15% following the quarterly report.",
    "Job BATCH001 has been running for 3 hours exceeding normal baseline."
]

def get_model_info() -> Dict[str, Dict]:
    """
    Information about different spaCy models
    
    Returns:
        Dictionary with model information
    """
    return {
        "en_core_web_sm": {
            "size": "15 MB",
            "vocab_size": "50K",
            "description": "Small model - Fast, basic accuracy",
            "use_case": "Development, testing, resource-constrained environments"
        },
        "en_core_web_md": {
            "size": "50 MB", 
            "vocab_size": "50K + 20K word vectors",
            "description": "Medium model - Balanced speed/accuracy",
            "use_case": "Production applications, good balance"
        },
        "en_core_web_lg": {
            "size": "750 MB",
            "vocab_size": "50K + 500K word vectors", 
            "description": "Large model - Slower, best accuracy",
            "use_case": "High-accuracy requirements, research"
        },
        "blank": {
            "size": "< 1 MB",
            "vocab_size": "Minimal",
            "description": "Blank model - Fastest, no pre-trained components",
            "use_case": "Custom pipelines, minimal processing"
        }
    }

def benchmark_model(model_name: str, texts: List[str], iterations: int = 5) -> Dict:
    """
    Benchmark a spaCy model's performance
    
    Args:
        model_name: Name of the model to benchmark
        texts: List of texts to process
        iterations: Number of iterations for timing
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        # Load model
        if model_name == "blank":
            nlp = spacy.blank("en")
        else:
            nlp = spacy.load(model_name)
        
        # Warm up
        for text in texts[:2]:
            nlp(text)
        
        # Benchmark processing speed
        processing_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            for text in texts:
                doc = nlp(text)
                # Force processing by accessing tokens
                _ = [token.text for token in doc]
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(processing_times)
        texts_per_second = len(texts) / avg_time
        
        # Test capabilities
        test_doc = nlp(texts[0])
        
        results = {
            "model_name": model_name,
            "status": "success",
            "avg_processing_time": avg_time,
            "texts_per_second": texts_per_second,
            "pipeline_components": nlp.pipe_names,
            "has_ner": nlp.has_pipe("ner"),
            "has_pos": nlp.has_pipe("tagger"),
            "has_parser": nlp.has_pipe("parser"),
            "has_vectors": nlp.has_pipe("tok2vec") or len(nlp.vocab.vectors) > 0,
            "vocab_size": len(nlp.vocab),
            "entities_found": len(test_doc.ents) if nlp.has_pipe("ner") else 0,
            "tokens_processed": len(test_doc)
        }
        
        return results
        
    except Exception as e:
        return {
            "model_name": model_name,
            "status": "error",
            "error": str(e),
            "avg_processing_time": float('inf'),
            "texts_per_second": 0
        }

def print_performance_comparison(results: List[Dict]):
    """
    Print formatted performance comparison
    
    Args:
        results: List of benchmark results
    """
    print("\n" + "="*80)
    print("🚀 spaCy Model Performance Comparison")
    print("="*80)
    
    # Sort by speed (fastest first)
    successful_results = [r for r in results if r["status"] == "success"]
    successful_results.sort(key=lambda x: x["avg_processing_time"])
    
    print(f"{'Model':<20} {'Speed':<15} {'Texts/sec':<12} {'Components':<25} {'Accuracy'}")
    print("-" * 80)
    
    for result in successful_results:
        model = result["model_name"]
        speed = f"{result['avg_processing_time']:.3f}s"
        tps = f"{result['texts_per_second']:.1f}"
        components = f"{len(result['pipeline_components'])}"
        
        # Accuracy indicators
        accuracy_score = 0
        if result["has_ner"]: accuracy_score += 1
        if result["has_pos"]: accuracy_score += 1  
        if result["has_parser"]: accuracy_score += 1
        if result["has_vectors"]: accuracy_score += 2
        
        accuracy = "⭐" * min(accuracy_score, 5)
        
        print(f"{model:<20} {speed:<15} {tps:<12} {components:<25} {accuracy}")

def print_detailed_analysis(results: List[Dict], model_info: Dict):
    """
    Print detailed analysis of each model
    
    Args:
        results: Benchmark results
        model_info: Model information dictionary
    """
    print("\n" + "="*60)
    print("📊 Detailed Model Analysis")
    print("="*60)
    
    for result in results:
        if result["status"] != "success":
            continue
            
        model = result["model_name"]
        info = model_info.get(model, {})
        
        print(f"\n🔍 {model.upper()}")
        print(f"   Size: {info.get('size', 'Unknown')}")
        print(f"   Description: {info.get('description', 'No description')}")
        print(f"   Processing Speed: {result['avg_processing_time']:.3f} seconds")
        print(f"   Throughput: {result['texts_per_second']:.1f} texts/second")
        print(f"   Pipeline: {', '.join(result['pipeline_components'])}")
        print(f"   Vocabulary: {result['vocab_size']:,} entries")
        print(f"   Use Case: {info.get('use_case', 'General purpose')}")

def get_recommendations() -> Dict[str, str]:
    """
    Get model recommendations for different use cases
    
    Returns:
        Dictionary of recommendations
    """
    return {
        "Development/Testing": "en_core_web_sm - Fast iteration, good enough accuracy",
        "Production Chatbot": "en_core_web_sm or en_core_web_md - Balance of speed and accuracy", 
        "High-Volume Processing": "en_core_web_sm or blank - Maximum throughput",
        "Research/Analysis": "en_core_web_lg - Best accuracy for detailed analysis",
        "Resource Constrained": "blank model - Minimal memory footprint",
        "Real-time Applications": "en_core_web_sm - Sub-second response times",
        "Batch Processing": "en_core_web_md - Good accuracy for offline processing"
    }

def main():
    """
    Main function to run performance comparison
    """
    print("🚀 Starting spaCy Model Performance Analysis...")
    
    # Models to test (only test available ones)
    models_to_test = ["blank", "en_core_web_sm"]
    
    # Try to add other models if available
    for model in ["en_core_web_md", "en_core_web_lg"]:
        try:
            spacy.load(model)
            models_to_test.append(model)
        except OSError:
            print(f"⚠️ {model} not available (not installed)")
    
    print(f"📋 Testing models: {', '.join(models_to_test)}")
    
    # Run benchmarks
    results = []
    for model in models_to_test:
        print(f"🧪 Benchmarking {model}...")
        result = benchmark_model(model, TEST_TEXTS)
        results.append(result)
    
    # Print results
    model_info = get_model_info()
    print_performance_comparison(results)
    print_detailed_analysis(results, model_info)
    
    # Print recommendations
    print("\n" + "="*60)
    print("💡 Model Recommendations by Use Case")
    print("="*60)
    recommendations = get_recommendations()
    for use_case, recommendation in recommendations.items():
        print(f"• {use_case}: {recommendation}")
    
    print("\n✅ Performance analysis complete!")
    
    return results

if __name__ == "__main__":
    main() 
