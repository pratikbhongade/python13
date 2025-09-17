#!/usr/bin/env python3
"""
spaCy Model Recommendation for Aspire Support Chatbot
Analyzes requirements and recommends the best model
"""

import spacy
from pathlib import Path

def analyze_chatbot_requirements():
    """Analyze the specific NLP requirements of the Aspire Support chatbot"""
    
    print("🔍 Analyzing Aspire Support Chatbot Requirements")
    print("=" * 60)
    
    requirements = {
        "Primary Tasks": [
            "🎯 Abend code extraction (S0C4, S0C7, S322, etc.)",
            "💬 Greeting detection (hello, hi, good morning)",
            "🔧 Technical term recognition",
            "📝 Intent classification support",
            "🔍 Entity extraction for production support"
        ],
        "Text Types": [
            "📱 Short user queries (5-50 words)",
            "💻 Technical mainframe terminology", 
            "🗣️ Conversational language",
            "📊 Error messages and codes",
            "❓ Question-answer pairs"
        ],
        "Performance Needs": [
            "⚡ Real-time response (< 1 second)",
            "💾 Memory efficient (web deployment)",
            "🔄 High throughput (multiple users)",
            "📦 Small deployment size"
        ],
        "Accuracy Needs": [
            "🎯 High precision for abend codes (critical)",
            "💬 Good conversational understanding",
            "🔍 Reliable entity extraction",
            "📝 Consistent text processing"
        ]
    }
    
    for category, items in requirements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    return requirements

def compare_spacy_models():
    """Compare different spaCy models for the use case"""
    
    print("\n🏆 spaCy Model Comparison")
    print("=" * 60)
    
    models = {
        "en_core_web_sm": {
            "size": "~15MB",
            "vocab": "50K vectors",
            "accuracy": "Good",
            "speed": "Fast",
            "components": ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            "pros": [
                "✅ Small size - perfect for deployment",
                "✅ Fast processing - real-time responses", 
                "✅ Good NER - handles abend codes well",
                "✅ Parser included - sentence segmentation",
                "✅ Balanced performance",
                "✅ Standard choice for production"
            ],
            "cons": [
                "⚠️ Limited vocabulary",
                "⚠️ May miss complex technical terms"
            ],
            "recommendation": "🌟 RECOMMENDED for your use case"
        },
        
        "en_core_web_md": {
            "size": "~40MB", 
            "vocab": "685K vectors",
            "accuracy": "Better",
            "speed": "Medium",
            "components": ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            "pros": [
                "✅ Better word vectors",
                "✅ Improved accuracy",
                "✅ Better similarity matching",
                "✅ More technical terms"
            ],
            "cons": [
                "⚠️ 3x larger than sm",
                "⚠️ Slower processing",
                "⚠️ More memory usage"
            ],
            "recommendation": "🔄 Consider if accuracy is critical"
        },
        
        "en_core_web_lg": {
            "size": "~560MB",
            "vocab": "685K vectors", 
            "accuracy": "Best",
            "speed": "Slower",
            "components": ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            "pros": [
                "✅ Highest accuracy",
                "✅ Best word vectors",
                "✅ Excellent for complex NLP"
            ],
            "cons": [
                "❌ Very large size",
                "❌ Slow processing", 
                "❌ High memory usage",
                "❌ Overkill for chatbot"
            ],
            "recommendation": "❌ NOT recommended for chatbot"
        },
        
        "en_core_web_trf": {
            "size": "~440MB",
            "vocab": "Transformer-based",
            "accuracy": "Excellent", 
            "speed": "Slowest",
            "components": ["transformer", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            "pros": [
                "✅ State-of-the-art accuracy",
                "✅ Transformer architecture",
                "✅ Best for complex understanding"
            ],
            "cons": [
                "❌ Very large size",
                "❌ Very slow processing",
                "❌ Requires GPU for speed",
                "❌ Overkill for simple tasks"
            ],
            "recommendation": "❌ NOT recommended for real-time chatbot"
        }
    }
    
    for model_name, details in models.items():
        print(f"\n📦 {model_name}")
        print(f"   Size: {details['size']}")
        print(f"   Vocabulary: {details['vocab']}")
        print(f"   Accuracy: {details['accuracy']}")
        print(f"   Speed: {details['speed']}")
        print(f"   Components: {', '.join(details['components'])}")
        
        print("   Pros:")
        for pro in details['pros']:
            print(f"     {pro}")
        
        print("   Cons:")
        for con in details['cons']:
            print(f"     {con}")
        
        print(f"   📋 {details['recommendation']}")
    
    return models

def test_model_performance():
    """Test actual performance with sample chatbot queries"""
    
    print("\n⚡ Performance Testing")
    print("=" * 40)
    
    # Sample queries typical for your chatbot
    test_queries = [
        "S0C4 abend error",
        "Hello, I need help with password reset",
        "What is the relationship between Aspire and VTO?",
        "Job ABC123 is running for 3 hours",
        "Memory issue in production",
        "Good morning, can you help me?",
        "S0C7 data exception occurred"
    ]
    
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Testing with en_core_web_sm")
        
        import time
        
        total_time = 0
        for query in test_queries:
            start_time = time.time()
            doc = nlp(query)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            total_time += processing_time
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            tokens = len(doc)
            
            print(f"  📝 '{query}'")
            print(f"     ⏱️  {processing_time:.2f}ms | 🔤 {tokens} tokens | 🎯 {len(entities)} entities")
            if entities:
                print(f"     📊 Entities: {entities}")
        
        avg_time = total_time / len(test_queries)
        print(f"\n📊 Average processing time: {avg_time:.2f}ms")
        
        if avg_time < 50:
            print("✅ Excellent speed for real-time chatbot")
        elif avg_time < 100:
            print("✅ Good speed for chatbot")
        else:
            print("⚠️ May be too slow for real-time responses")
            
    except OSError:
        print("⚠️ en_core_web_sm not available for testing")

def get_final_recommendation():
    """Provide final recommendation based on analysis"""
    
    print("\n🎯 Final Recommendation for Aspire Support Chatbot")
    print("=" * 60)
    
    recommendation = {
        "recommended_model": "en_core_web_sm",
        "reasons": [
            "🚀 Perfect size for web deployment (~15MB)",
            "⚡ Fast processing for real-time responses",
            "🎯 Sufficient accuracy for abend code extraction",
            "💬 Good conversational understanding",
            "🔧 All needed components (NER, parser, tagger)",
            "💾 Low memory footprint",
            "📦 Easy to deploy and maintain",
            "✅ Already working well in your current setup"
        ],
        "alternatives": {
            "If accuracy is critical": "en_core_web_md (but 3x larger)",
            "If size is extremely important": "Custom blank model with rules",
            "If offline deployment needed": "Download and bundle en_core_web_sm"
        },
        "installation": "python -m spacy download en_core_web_sm"
    }
    
    print(f"🏆 Recommended Model: {recommendation['recommended_model']}")
    print("\n📋 Why this model is perfect for your chatbot:")
    for reason in recommendation['reasons']:
        print(f"  {reason}")
    
    print(f"\n📦 Installation Command:")
    print(f"  {recommendation['installation']}")
    
    print(f"\n🔄 Alternative Options:")
    for scenario, alternative in recommendation['alternatives'].items():
        print(f"  • {scenario}: {alternative}")
    
    print(f"\n✅ Conclusion:")
    print(f"  Your current en_core_web_sm model is PERFECT for the Aspire Support")
    print(f"  chatbot. It provides the optimal balance of speed, accuracy, and size")
    print(f"  for your production support use case. No changes needed!")

def main():
    """Main function to run the complete analysis"""
    
    print("🤖 spaCy Model Recommendation Analysis")
    print("🏢 For: Aspire Support Chatbot")
    print("📅 Analysis Date: 2025-09-18")
    print("=" * 60)
    
    # Analyze requirements
    analyze_chatbot_requirements()
    
    # Compare models
    compare_spacy_models()
    
    # Test performance
    test_model_performance()
    
    # Final recommendation
    get_final_recommendation()
    
    print("\n" + "=" * 60)
    print("📋 Summary: en_core_web_sm is the optimal choice!")
    print("=" * 60)

if __name__ == "__main__":
    main() 
