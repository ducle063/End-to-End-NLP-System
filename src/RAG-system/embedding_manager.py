import torch
import logging
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

from config import EmbeddingConfig, DatabaseConfig

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, embedding_config: EmbeddingConfig, db_config: DatabaseConfig):
        self.embedding_config = embedding_config
        self.db_config = db_config
        self.embedding_model = None
        self.vector_db = None
        self.setup_embeddings()
        self.setup_database()

    def setup_embeddings(self):
        """Initialize embedding model"""
        try:
            # Determine device
            if self.embedding_config.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.embedding_config.device)

            encode_kwargs = {
                'normalize_embeddings': self.embedding_config.normalize_embeddings
            }

            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.embedding_config.model_name,
                model_kwargs={'device': device},
                encode_kwargs=encode_kwargs
            )

            logger.info(f"Initialized embedding model: {self.embedding_config.model_name} on {device}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def setup_database(self):
        """Initialize vector database"""
        try:
            self.vector_db = Chroma(
                persist_directory=self.db_config.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.db_config.collection_name
            )

            logger.info(f"Initialized vector database at: {self.db_config.persist_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    def create_retriever(self, search_k: int = 100, filters: Optional[Dict[str, Any]] = None):
        """Create retriever with specified search parameters and filters"""
        if not self.vector_db:
            raise RuntimeError("Vector database not initialized")

        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": search_k, "filter": filters} if filters else {"k": search_k}
        )

        logger.info(f"Created retriever with k={search_k}, filters={filters}")
        return retriever

    def add_documents(self, documents: List[Document]):
        """Add documents to vector database"""
        try:
            self.vector_db.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector database")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search with optional filters"""
        try:
            docs = self.vector_db.similarity_search(query, k=k, filter=filters)
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            collection = self.vector_db._collection
            count = collection.count()
            return {
                "document_count": count,
                "collection_name": self.db_config.collection_name,
                "persist_directory": self.db_config.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage (requires your config to be set up)
    from config import EmbeddingConfig, DatabaseConfig

    embedding_config = EmbeddingConfig(
        model_name='BAAI/bge-m3',
        device='auto',
        normalize_embeddings=True
    )
    db_config = DatabaseConfig(
        persist_directory='vdb',
        collection_name='documents'
    )

    embedding_manager = EmbeddingManager(embedding_config, db_config)

    # Example query with filter
    query = "Hiệu trưởng trường Đại học Công nghệ là ai?"
    filters = {"university": "uet", "category": "gioi_thieu"}
    results = embedding_manager.similarity_search(query, k=3)
    print("\nSearch results with filter:")
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)

    # # Example query without filter
    # query_all = "Thông báo mới nhất?"
    # all_results = embedding_manager.similarity_search(query_all, k=3)
    # print("\nSearch results without filter:")
    # for doc in all_results:
    #     print(doc.page_content)
    #     print(doc.metadata)

    print("\nDatabase stats:", embedding_manager.get_database_stats())