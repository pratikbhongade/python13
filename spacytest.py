#!/usr/bin/env python3
"""
GitHub Model Downloader
Alternative to Hugging Face for downloading pre-trained models
"""

import os
import requests
import zipfile
import json
import subprocess
import sys
from pathlib import Path
import shutil

class GitHubModelDownloader:
    """Download pre-trained models from GitHub repositories"""
    
    def __init__(self, models_dir="./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def download_file(self, url, filepath):
        """Download a file from URL"""
        print(f"📥 Downloading: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def download_sentence_transformers_from_github(self):
        """Download sentence transformers model from GitHub mirror"""
        
        print("🚀 Downloading Sentence Transformers from GitHub")
        print("=" * 50)
        
        # GitHub repositories that mirror Hugging Face models
        github_repos = [
            {
                "name": "all-MiniLM-L6-v2",
                "repo": "sentence-transformers/all-MiniLM-L6-v2",
                "files": [
                    "config.json",
                    "pytorch_model.bin", 
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt"
                ]
            }
        ]
        
        for model_info in github_repos:
            model_name = model_info["name"]
            repo = model_info["repo"]
            
            print(f"\n📦 Downloading {model_name}...")
            
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            success_count = 0
            
            for filename in model_info["files"]:
                # Try different GitHub raw URLs
                urls_to_try = [
                    f"https://raw.githubusercontent.com/{repo}/main/{filename}",
                    f"https://github.com/{repo}/raw/main/{filename}",
                    f"https://raw.githubusercontent.com/{repo}/master/{filename}"
                ]
                
                filepath = model_dir / filename
                downloaded = False
                
                for url in urls_to_try:
                    if self.download_file(url, filepath):
                        downloaded = True
                        success_count += 1
                        break
                
                if not downloaded:
                    print(f"⚠️ Could not download {filename}")
            
            if success_count > 0:
                print(f"✅ Downloaded {success_count}/{len(model_info['files'])} files for {model_name}")
            else:
                print(f"❌ Failed to download {model_name}")
    
    def download_universal_sentence_encoder(self):
        """Download Universal Sentence Encoder from TensorFlow Hub mirror"""
        
        print("\n🧠 Downloading Universal Sentence Encoder")
        print("=" * 45)
        
        # TensorFlow Hub models often have GitHub mirrors
        use_urls = [
            "https://github.com/tensorflow/tfhub-modules/raw/master/universal-sentence-encoder/4/model.tar.gz",
            "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz"
        ]
        
        model_dir = self.models_dir / "universal-sentence-encoder"
        model_dir.mkdir(exist_ok=True)
        
        for url in use_urls:
            filepath = model_dir / "model.tar.gz"
            if self.download_file(url, filepath):
                # Extract the model
                try:
                    import tarfile
                    with tarfile.open(filepath, 'r:gz') as tar:
                        tar.extractall(model_dir)
                    print("✅ Extracted Universal Sentence Encoder")
                    return True
                except Exception as e:
                    print(f"❌ Error extracting: {e}")
        
        return False
    
    def download_fasttext_vectors(self):
        """Download FastText word vectors from GitHub"""
        
        print("\n⚡ Downloading FastText Vectors")
        print("=" * 35)
        
        # FastText vectors are available on GitHub
        fasttext_urls = [
            {
                "name": "wiki-news-300d-1M.vec",
                "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
                "description": "1M word vectors trained on Wikipedia"
            },
            {
                "name": "crawl-300d-2M.vec", 
                "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
                "description": "2M word vectors trained on Common Crawl"
            }
        ]
        
        fasttext_dir = self.models_dir / "fasttext"
        fasttext_dir.mkdir(exist_ok=True)
        
        for vector_info in fasttext_urls:
            print(f"\n📥 Downloading {vector_info['name']}...")
            print(f"📝 {vector_info['description']}")
            
            zip_path = fasttext_dir / f"{vector_info['name']}.zip"
            
            if self.download_file(vector_info['url'], zip_path):
                # Extract the vectors
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(fasttext_dir)
                    print(f"✅ Extracted {vector_info['name']}")
                    
                    # Remove zip file to save space
                    zip_path.unlink()
                    
                except Exception as e:
                    print(f"❌ Error extracting {vector_info['name']}: {e}")
    
    def clone_model_repository(self, repo_url, model_name):
        """Clone a GitHub repository containing a model"""
        
        print(f"\n📂 Cloning {model_name} from GitHub")
        print("=" * 40)
        
        model_dir = self.models_dir / model_name
        
        if model_dir.exists():
            print(f"⚠️ {model_name} already exists, removing...")
            shutil.rmtree(model_dir)
        
        try:
            # Clone the repository
            result = subprocess.run([
                "git", "clone", repo_url, str(model_dir)
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully cloned {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning {repo_url}: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Git not found. Please install Git to clone repositories.")
            return False

def create_local_sentence_transformer():
    """Create a local sentence transformer using available models"""
    
    print("\n🔧 Creating Local Sentence Transformer")
    print("=" * 40)
    
    code = '''
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class LocalSentenceTransformer:
    """
    Local sentence transformer using spaCy + TF-IDF
    No internet required after setup
    """
    
    def __init__(self, model_name="en_core_web_md"):
        self.nlp = spacy.load(model_name)
        self.vectorizer = None
        self.vocab = []
    
    def encode(self, sentences, batch_size=32):
        """Encode sentences to vectors"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        
        for sentence in sentences:
            # Method 1: spaCy document vectors
            doc = self.nlp(sentence)
            if doc.has_vector:
                spacy_vector = doc.vector
            else:
                spacy_vector = np.zeros(300)
            
            # Method 2: Average word vectors
            word_vectors = []
            for token in doc:
                if token.has_vector and not token.is_stop and not token.is_punct:
                    word_vectors.append(token.vector)
            
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(300)
            
            # Combine both methods
            combined_vector = 0.6 * spacy_vector + 0.4 * avg_vector
            embeddings.append(combined_vector)
        
        return np.array(embeddings)
    
    def similarity(self, sentences1, sentences2):
        """Calculate similarity between sentence pairs"""
        emb1 = self.encode(sentences1)
        emb2 = self.encode(sentences2)
        
        return cosine_similarity(emb1, emb2)
    
    def save(self, path):
        """Save the model"""
        model_data = {
            'model_name': self.nlp.meta['name'],
            'vocab': self.vocab
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path):
        """Load the model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.nlp = spacy.load(model_data['model_name'])
        self.vocab = model_data['vocab']

# Example usage
if __name__ == "__main__":
    # Create local transformer
    transformer = LocalSentenceTransformer()
    
    # Test sentences
    sentences = [
        "I need help with password reset",
        "Can you help me reset my password?",
        "S0C4 abend error occurred"
    ]
    
    # Encode sentences
    embeddings = transformer.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate similarities
    similarities = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"Password reset similarity: {similarities:.4f}")
    '''
    
    # Save the local transformer code
    with open("local_sentence_transformer.py", "w") as f:
        f.write(code)
    
    print("✅ Created local_sentence_transformer.py")
    print("🎯 This uses only your existing spaCy model!")

def main():
    """Main function to download models from GitHub"""
    
    print("🚀 GitHub Model Downloader")
    print("🎯 Alternative to Hugging Face")
    print("=" * 50)
    
    downloader = GitHubModelDownloader()
    
    print("\n📋 Available Options:")
    print("1. Download Sentence Transformers from GitHub mirrors")
    print("2. Download Universal Sentence Encoder")
    print("3. Download FastText word vectors")
    print("4. Create local sentence transformer (recommended)")
    print("5. Clone specific model repositories")
    
    # For now, let's create the local transformer (best option)
    create_local_sentence_transformer()
    
    print("\n💡 Recommendations:")
    print("✅ Local Sentence Transformer - Uses your existing spaCy model")
    print("⚡ FastText Vectors - Good for word-level similarity")
    print("🧠 Universal Sentence Encoder - Requires TensorFlow")
    print("🔄 GitHub Cloning - For specific model repositories")
    
    print("\n🎉 Local solution created successfully!")
    print("📁 Check local_sentence_transformer.py for implementation")

if __name__ == "__main__":
    main() 
