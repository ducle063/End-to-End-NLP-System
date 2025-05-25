import os
from typing import Dict, List
import re
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib

class ReferenceDocumentProcessor:
    def __init__(self, embedding_model_name='BAAI/bge-m3'):
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.client = chromadb.PersistentClient(path="vdb")
        self.collection = self.client.get_or_create_collection(name="documents",)
    
    def parse_document(self, file_path: str) -> Dict:
        """Parse a single document file into structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        parts = content.split('\n\n', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid document format in {file_path}")
        
        metadata_lines = parts[0].split('\n')
        text = parts[1].strip()
        
        metadata = {}
        for line in metadata_lines:
            if line.startswith('URL:'):
                metadata['url'] = line[4:].strip()
            elif line.startswith('Title:'):
                metadata['title'] = line[6:].strip()
        
        if 'title' not in metadata:
            raise ValueError(f"Missing required 'Title:' metadata in {file_path}")
        
        # Nếu không có URL, thay bằng 'unknown'
        if 'url' not in metadata:
            metadata['url'] = 'unknown'
        
        return {
            'url': metadata['url'],
            'title': metadata['title'],
            'text': text,
            'source': os.path.basename(file_path)
        }

    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 200) -> List[Dict]:
        """Split text into chunks with overlap"""
        words = re.split(r'\s+', text)
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunks.append({
                'text': chunk_text,
                'start_word': start,
                'end_word': end,
                'chunk_size': len(chunk_text)
            })
            
            start = end - overlap if end - overlap > start else end
        
        return chunks
    
    def generate_id(self, source: str, chunk_index: int, chunk_text: str) -> str:
        """Generate a unique ID for each chunk"""
        unique_str = f"{source}_{chunk_index}_{chunk_text[:50]}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def process_folder(self, folder_path: str, batch_size: int = 32):
        """Process all documents in a folder"""
        documents = []
        metadatas = []
        ids = []
        
        for filename in os.listdir(folder_path):
            if not filename.endswith('.txt'):
                continue
                
            try:
                file_path = os.path.join(folder_path, filename)
                doc_data = self.parse_document(file_path)
                chunks = self.chunk_text(doc_data['text'])
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk['text'])
                    metadatas.append({
                        'url': doc_data['url'],
                        'title': doc_data['title'],
                        'source': doc_data['source'],
                        'chunk_index': i,
                        'start_word': chunk['start_word'],
                        'end_word': chunk['end_word']
                    })
                    ids.append(self.generate_id(doc_data['source'], i, chunk['text']))
                
                print(f"Processed {filename} with {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch_docs, convert_to_tensor=False)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_docs,
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        
        print(f"Processing complete. Added {len(documents)} chunks to the vector database.")

# Usage
if __name__ == "__main__":
    processor = ReferenceDocumentProcessor()
    processor.process_folder("/workspaces/End-to-End-NLP-System/data/docs/cleaned-docs")