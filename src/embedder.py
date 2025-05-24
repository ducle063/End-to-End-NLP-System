# src/embedder.py
import os
import hashlib
import json
from typing import Dict, List, Tuple, Any
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    pass