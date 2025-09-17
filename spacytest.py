#!/usr/bin/env python3
"""
Comprehensive spaCy Model Loader
Handles model loading with multiple fallback options and automatic installation
"""

import spacy
import subprocess
import sys
import logging
import os
from pathlib import Path

def install_spacy_model(model_name="en_core_web_sm"):
    """
    Automatically install spaCy model if not found
    
    Args:
        model_name: Name of the spaCy model to install
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        print(f"📦 Installing spaCy model: {model_name}")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", model_name
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ Successfully installed {model_name}")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {model_name}: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error installing {model_name}: {e}")
        return False

def load_spacy_model_with_fallbacks(preferred_model="en_core_web_sm"):
    """
    Load spaCy model with multiple fallback options
    
    Args:
        preferred_model: Preferred model name to load
        
    Returns:
        spacy.Language: Loaded spaCy model
    """
    
    # Method 1: Try loading the preferred model
    try:
        nlp = spacy.load(preferred_model)
        logging.info(f"✅ Successfully loaded {preferred_model}")
        return nlp
    except OSError:
        logging.warning(f"⚠️ Could not load {preferred_model}")
    
    # Method 2: Try loading from local directory
    local_paths = [
        f"./{preferred_model}",
        f"./model/{preferred_model}",
        f"../models/{preferred_model}",
        preferred_model.replace("_", "-")  # Try with hyphens
    ]
    
    for path in local_paths:
        try:
            nlp = spacy.load(path)
            logging.info(f"✅ Successfully loaded model from: {path}")
            return nlp
        except OSError:
            continue
    
    # Method 3: Try alternative model names
    alternative_models = [
        "en_core_web_md",  # Medium model
        "en_core_web_lg",  # Large model
        "en",              # Generic English
    ]
    
    for model in alternative_models:
        try:
            nlp = spacy.load(model)
            logging.info(f"✅ Successfully loaded alternative model: {model}")
            return nlp
        except OSError:
            continue
    
    # Method 4: Attempt automatic installation
    logging.info(f"🔄 Attempting to install {preferred_model}...")
    if install_spacy_model(preferred_model):
        try:
            nlp = spacy.load(preferred_model)
            logging.info(f"✅ Successfully loaded {preferred_model} after installation")
            return nlp
        except OSError:
            logging.error(f"❌ Still cannot load {preferred_model} after installation")
    
    # Method 5: Create blank English model as last resort
    logging.warning("⚠️ Using blank English model as fallback")
    nlp = spacy.blank("en")
    
    # Add basic components to blank model
    try:
        nlp.add_pipe("sentencizer")
        logging.info("✅ Added sentencizer to blank model")
    except Exception as e:
        logging.warning(f"Could not add sentencizer: {e}")
    
    return nlp

def check_model_capabilities(nlp):
    """
    Check what capabilities the loaded model has
    
    Args:
        nlp: Loaded spaCy model
        
    Returns:
        dict: Dictionary of model capabilities
    """
    capabilities = {
        "has_ner": nlp.has_pipe("ner"),
        "has_tagger": nlp.has_pipe("tagger"),
        "has_parser": nlp.has_pipe("parser"),
        "has_lemmatizer": nlp.has_pipe("lemmatizer"),
        "has_sentencizer": nlp.has_pipe("sentencizer"),
        "pipeline": nlp.pipe_names,
        "vocab_size": len(nlp.vocab),
        "model_name": nlp.meta.get("name", "unknown"),
        "model_version": nlp.meta.get("version", "unknown")
    }
    
    return capabilities

def print_model_info(nlp):
    """
    Print detailed information about the loaded model
    
    Args:
        nlp: Loaded spaCy model
    """
    capabilities = check_model_capabilities(nlp)
    
    print("\n" + "="*50)
    print("🤖 spaCy Model Information")
    print("="*50)
    print(f"Model Name: {capabilities['model_name']}")
    print(f"Model Version: {capabilities['model_version']}")
    print(f"Vocabulary Size: {capabilities['vocab_size']:,}")
    print(f"Pipeline Components: {', '.join(capabilities['pipeline'])}")
    print("\nCapabilities:")
    print(f"  • Named Entity Recognition: {'✅' if capabilities['has_ner'] else '❌'}")
    print(f"  • Part-of-Speech Tagging: {'✅' if capabilities['has_tagger'] else '❌'}")
    print(f"  • Dependency Parsing: {'✅' if capabilities['has_parser'] else '❌'}")
    print(f"  • Lemmatization: {'✅' if capabilities['has_lemmatizer'] else '❌'}")
    print(f"  • Sentence Segmentation: {'✅' if capabilities['has_sentencizer'] else '❌'}")
    print("="*50)

# Example usage functions
def simple_model_loading():
    """Simple model loading example"""
    try:
        # Basic loading
        nlp = spacy.load("en_core_web_sm")
        print("✅ Model loaded successfully!")
        return nlp
    except OSError:
        print("❌ Model not found. Please install it:")
        print("python -m spacy download en_core_web_sm")
        return None

def robust_model_loading():
    """Robust model loading with error handling"""
    models_to_try = ["en_core_web_sm", "en_core_web_md", "en"]
    
    for model_name in models_to_try:
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Successfully loaded: {model_name}")
            return nlp
        except OSError:
            print(f"⚠️ Could not load: {model_name}")
            continue
    
    # Fallback to blank model
    print("⚠️ Using blank English model")
    return spacy.blank("en")

def main():
    """
    Main function to demonstrate model loading
    """
    print("🚀 spaCy Model Loading Examples")
    print("="*40)
    
    # Load model with comprehensive fallbacks
    nlp = load_spacy_model_with_fallbacks()
    
    # Print model information
    print_model_info(nlp)
    
    # Test the model
    test_text = "Hello, this is a test sentence for spaCy processing."
    doc = nlp(test_text)
    
    print(f"\n📝 Test Processing:")
    print(f"Input: {test_text}")
    print(f"Tokens: {[token.text for token in doc]}")
    
    if nlp.has_pipe("ner"):
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Entities: {entities}")
    
    return nlp

if __name__ == "__main__":
    nlp = main() 
