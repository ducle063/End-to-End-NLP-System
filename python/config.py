import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Device detection with fallback
    USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"
    DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
    
    # Model Settings (with smaller defaults)
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    GENERATOR_MODEL = "vinai/phobert-base"
    
    # Vector Database
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vietnamese_docs")
    
    # Performance
    MAX_CONTEXT_LENGTH = 2000
    MODEL_LOADED = False  # Track if models are loaded