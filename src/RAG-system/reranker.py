import logging
from typing import List, Tuple, Any
from langchain.schema import Document

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("FlagEmbedding not available. Reranking will be disabled.")

from config import RerankerConfig

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.reranker = None
        self.setup_reranker()
    
    def setup_reranker(self):
        """Initialize reranker model"""
        if not RERANKER_AVAILABLE:
            logger.warning("Reranker not available - will return documents in original order")
            return
        
        try:
            self.reranker = FlagReranker(
                self.config.model_name, 
                use_fp16=self.config.use_fp16
            )
            logger.info(f"Initialized reranker: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """Rerank documents based on relevance to query"""
        if not self.reranker or not documents:
            logger.info("Reranker not available or no documents - returning original order")
            return documents[:top_k]
        
        try:
            # Prepare documents for reranking
            docs_for_reranking = []
            for doc in documents:
                docs_for_reranking.append([query, str(doc)])
            
            # Compute reranking scores
            scores = self.reranker.compute_score(docs_for_reranking)
            
            # Handle single document case
            if not isinstance(scores, list):
                scores = [scores]
            
            # Combine documents with scores
            doc_score_pairs = list(zip(documents, scores))
            
            # Sort by score (descending)
            sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            
            # Extract top-k documents
            reranked_docs = [doc for doc, score in sorted_pairs[:top_k]]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            
            # Log scores for debugging
            for i, (doc, score) in enumerate(sorted_pairs[:top_k]):
                logger.debug(f"Rank {i+1}: Score {score:.4f}")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    def is_available(self) -> bool:
        """Check if reranker is available and functional"""
        return self.reranker is not None
    
    def get_scores(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Get documents with their reranking scores"""
        if not self.reranker or not documents:
            return [(doc, 0.0) for doc in documents]
        
        try:
            docs_for_reranking = [[query, str(doc)] for doc in documents]
            scores = self.reranker.compute_score(docs_for_reranking)
            
            if not isinstance(scores, list):
                scores = [scores]
            
            return list(zip(documents, scores))
            
        except Exception as e:
            logger.error(f"Failed to get reranking scores: {e}")
            return [(doc, 0.0) for doc in documents]