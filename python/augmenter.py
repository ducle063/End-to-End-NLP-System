from config import Config
from pyvi import ViTokenizer
import re

class ContextAugmenter:
    def __init__(self):
        self.config = Config()
    
    def clean_text(self, text: str) -> str:
        """Basic Vietnamese text cleaning"""
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return ViTokenizer.tokenize(text)  # Vietnamese word segmentation
    
    def augment(self, documents: list, max_length: int = None) -> str:
        """Simple concatenation with cleaning"""
        if max_length is None:
            max_length = self.config.MAX_CONTEXT_LENGTH
            
        cleaned_docs = []
        for doc in documents:
            try:
                cleaned = self.clean_text(doc)[:1000]  # Truncate long documents
                cleaned_docs.append(cleaned)
            except:
                cleaned_docs.append(doc[:1000])
        
        context = "\n\n".join(
            f"[Document {i+1}]: {doc}" 
            for i, doc in enumerate(cleaned_docs))
        
        return context[:max_length]