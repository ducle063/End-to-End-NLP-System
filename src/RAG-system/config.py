import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto", "cuda", "cpu"

@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = True

@dataclass
class RetrieverConfig:
    search_k: int = 25
    top_k_after_rerank: int = 3

@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 150
    temperature: float = 0.7
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required")

@dataclass
class DatabaseConfig:
    persist_directory: str = "/workspaces/End-to-End-NLP-System/src/RAG-system/vdb"
    collection_name: str = "documents"

@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Pipeline settings
    use_reranking: bool = True
    use_multi_query: bool = True
    use_few_shot: bool = False  # Disabled by default as noted in original code
    
    # Rate limiting
    request_delay: float = 5.0  # seconds between requests