import chromadb
from chromadb.utils import embedding_functions
from config import Config
import warnings
warnings.filterwarnings("ignore")

class DocumentRetriever:
    def __init__(self):
        self.config = Config()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.EMBEDDING_MODEL,
            device=self.config.DEVICE
        )
        self.client = chromadb.PersistentClient(path=self.config.VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.ef
        )
    
    def retrieve(self, query: str, top_k: int = 3):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "ids": results["ids"][0]
            }
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return {"documents": [], "metadatas": [], "ids": []}