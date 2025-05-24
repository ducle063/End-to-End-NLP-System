from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
import logging
import re

class Retriever:
    def __init__(self, embedding_model_name='pending', 
                 reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 persist_directory="./vector_db", 
                 collection_name="reference_docs"):
        # Khởi tạo logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Khởi tạo mô hình nhúng và reranker
        self.model = SentenceTransformer(embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Sử dụng cosine similarity
        )

    def preprocess_text(self, text):
        """Tiền xử lý văn bản: chuyển thành chữ thường và loại bỏ ký tự đặc biệt."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Loại bỏ ký tự đặc biệt
        return ' '.join(text.split())  # Chuẩn hóa khoảng trắng

    def expand_query(self, query):
        """Mở rộng truy vấn với từ đồng nghĩa."""
        synonyms = {
            "thành lập": ["ra đời", "bắt đầu", "lịch sử", "khởi đầu"],
            "trường đại học công nghệ": ["UET", "Đại học Công nghệ", "ĐHQGHN"]
        }
        expanded_queries = [query]
        for key, values in synonyms.items():
            if key.lower() in query.lower():
                for synonym in values:
                    expanded_query = query.replace(key, synonym)
                    expanded_queries.append(expanded_query)
        return list(set(expanded_queries))  # Loại bỏ trùng lặp

    def retrieve(self, query, top_k=15, filter_metadata=None):
        """Truy xuất tài liệu với reranking."""
        try:
            # Tiền xử lý và mở rộng truy vấn
            processed_query = self.preprocess_text(query)
            expanded_queries = self.expand_query(query)
            self.logger.info(f"Original query: {query}")
            self.logger.info(f"Expanded queries: {expanded_queries}")

            # Mã hóa tất cả các truy vấn mở rộng
            query_embeddings = self.model.encode(expanded_queries, convert_to_tensor=False).tolist()

            # Truy vấn ChromaDB
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["documents", "metadatas"],
                where=filter_metadata  # Lọc theo metadata nếu có
            )

            # Lấy danh sách tài liệu và metadata
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []

            if not documents:
                self.logger.warning("No documents found for the query.")
                return []

            # Reranking với cross-encoder
            query_doc_pairs = [(query, doc) for doc in documents]
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Sắp xếp tài liệu theo điểm reranking
            ranked_docs = [
                (doc, meta, score)
                for doc, meta, score in sorted(
                    zip(documents, metadatas, rerank_scores),
                    key=lambda x: x[2],
                    reverse=True
                )
            ]

            # Trả về top_k tài liệu sau khi rerank
            final_docs = [
                {"document": doc, "metadata": meta, "score": score}
                for doc, meta, score in ranked_docs[:top_k]
            ]

            self.logger.info(f"Retrieved {len(final_docs)} documents after reranking.")
            return final_docs

        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            return []

if __name__ == '__main__':
    # Cấu hình metadata để lọc tài liệu liên quan đến lịch sử
    filter_metadata = {"category": {"$in": ["history", "introduction", "about"]}}

    retriever = Retriever()
    query = "Trường Đại học Công Nghệ được thành lập vào năm nào?"
    relevant_docs = retriever.retrieve(query, filter_metadata=filter_metadata)
    
    print(f"Query: {query}")
    print("Relevant documents:")
    for item in relevant_docs:
        print(f"- Document: {item['document']}")
        print(f"  Metadata: {item['metadata']}")
        print(f"  Score: {item['score']}")