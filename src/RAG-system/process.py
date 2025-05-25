import os
from typing import Dict, List
import re
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib

class ReferenceDocumentProcessor:
    def __init__(self, embedding_model_name='BAAI/bge-m3'):
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.client = chromadb.PersistentClient(path="/workspaces/End-to-End-NLP-System/src/RAG-system/vdb")
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
            elif line.startswith('TITLE:'):
                metadata['title'] = line[6:].strip()
            elif line.startswith('UNIVERSITY:'):
                metadata['university'] = line[11:].strip().lower()
            elif line.startswith('CATEGORY:'):
                metadata['category'] = line[9:].strip().lower() # Lấy category từ file
            elif line.startswith('MENU_TEXT:'):
                metadata['menu_text'] = line[10:].strip()

        if 'title' not in metadata:
            raise ValueError(f"Missing required 'Title:' metadata in {file_path}")

        if 'url' not in metadata:
            metadata['url'] = 'unknown'

        # Nếu không có category trong file, dùng category từ thư mục
        if 'category' not in metadata:
            metadata['category'] = "UNCATEGORIZED"
        if 'menu_text' not in metadata:
            metadata['menu_text'] = 'unknown'
        if not text:
            raise ValueError(f"Empty text content in {file_path}")
        

        return {
            'url': metadata['url'],
            'title': metadata['title'],
            'text': text,
            'source': os.path.basename(file_path),
            'university': metadata.get('university', os.path.basename(os.path.dirname(os.path.dirname(file_path))).lower()), # Lấy tên trường từ thư mục cha
            'category': metadata['category'],
            'menu_text': metadata.get('menu_text', 'unknown')
        }

    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 150) -> List[Dict]:
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
    
    def process_folder(self, root_folder: str, batch_size: int = 32):
        """Process all documents in the root folder and its subfolders"""
        all_documents = []
        all_metadatas = []
        all_ids = []

        for university_folder in os.listdir(root_folder):
            university_path = os.path.join(root_folder, university_folder)
            if os.path.isdir(university_path):
                for category_folder in os.listdir(university_path):
                    category_path = os.path.join(university_path, category_folder)
                    if os.path.isdir(category_path):
                        print(f"Processing university: {university_folder}, category: {category_folder}")
                        for filename in os.listdir(category_path):
                            if filename.endswith('.txt'):
                                try:
                                    file_path = os.path.join(category_path, filename)
                                    doc_data = self.parse_document(file_path)
                                    chunks = self.chunk_text(doc_data['text'])

                                    for i, chunk in enumerate(chunks):
                                        all_documents.append(chunk['text'])
                                        all_metadatas.append({
                                            'url': doc_data['url'],
                                            'title': doc_data['title'],
                                            'source': doc_data['source'],
                                            'chunk_index': i,
                                            'start_word': chunk['start_word'],
                                            'end_word': chunk['end_word'],
                                            'university': doc_data['university'],
                                            'category': doc_data['category'],
                                            'menu_text': doc_data['menu_text']
                                        })
                                        all_ids.append(self.generate_id(doc_data['source'], i, chunk['text']))

                                    print(f"  Processed {filename} with {len(chunks)} chunks")

                                except Exception as e:
                                    print(f"  Error processing {filename}: {str(e)}")

        # Process in batches
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size] # Lấy metadata tương ứng với batch
            batch_ids = all_ids[i:i+batch_size] # Lấy ids tương ứng với batch
            embeddings = self.embedding_model.encode(batch_docs, convert_to_tensor=False)

            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"Đã thêm batch {i // batch_size + 1} vào ChromaDB.")
            except Exception as e:
                print(f"Lỗi khi thêm batch {i // batch_size + 1} vào ChromaDB: {e}")
                break # Dừng vòng lặp nếu có lỗi

        print(f"Processing complete. Added {len(all_documents)} chunks to the vector database.")

# Usage
if __name__ == "__main__":
    processor = ReferenceDocumentProcessor()
    processor.process_folder("/workspaces/End-to-End-NLP-System/data/clean/extra-info")