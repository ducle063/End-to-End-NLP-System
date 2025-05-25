import time
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from config import RAGConfig
from gemini_client import GeminiClient
from embedding_manager import EmbeddingManager
from reranker import DocumentReranker
from prompt_manager import PromptManager
from utils import unique_documents, setup_logging, format_answer

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_components()
    
    def setup_components(self):
        """Initialize all pipeline components"""
        try:
            # Setup logging
            setup_logging()
            
            # Initialize components
            self.llm_client = GeminiClient(self.config.gemini)
            self.embedding_manager = EmbeddingManager(
                self.config.embedding, 
                self.config.database
            )
            
            if self.config.use_reranking:
                self.reranker = DocumentReranker(self.config.reranker)
            else:
                self.reranker = None
            
            self.prompt_manager = PromptManager()
            
            # Create retriever
            self.retriever = self.embedding_manager.create_retriever(
                search_k=self.config.retriever.search_k
            )
            
            logger.info("RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents if reranking is enabled"""
        if not self.config.use_reranking or not self.reranker:
            return documents[:self.config.retriever.top_k_after_rerank]
        
        return self.reranker.rerank_documents(
            query, documents, self.config.retriever.top_k_after_rerank
        )
    
    def generate_paraphrased_queries(self, original_query: str) -> List[str]:
        """Generate paraphrased versions of the original query"""
        if not self.config.use_multi_query:
            return []
        
        try:
            paraphrased = self.llm_client.generate_paraphrased_questions(original_query)
            logger.info(f"Generated {len(paraphrased)} paraphrased queries")
            return paraphrased
        except Exception as e:
            logger.error(f"Failed to generate paraphrased queries: {e}")
            return []
    
    def multi_query_retrieval(self, original_query: str) -> List[Document]:
        """Perform multi-query retrieval and combine results"""
        all_docs = []
        
        # Get paraphrased queries
        paraphrased_queries = self.generate_paraphrased_queries(original_query)
        
        # Retrieve documents for each paraphrased query
        for query in paraphrased_queries:
            if query.strip():
                docs = self.retrieve_documents(query)
                all_docs.extend(docs)
        
        # Also retrieve documents for original query
        original_docs = self.retrieve_documents(original_query)
        all_docs.extend(original_docs)
        
        # Remove duplicates
        unique_docs = unique_documents(all_docs)
        
        logger.info(f"Multi-query retrieval: {len(all_docs)} total, {len(unique_docs)} unique documents")
        return unique_docs
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the LLM"""
        try:
            prompt = self.prompt_manager.build_qa_prompt(
                question=query,
                context=context,
                use_few_shot=self.config.use_few_shot
            )
            
            answer = self.llm_client.generate_answer(prompt)
            return format_answer(answer)
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Failed to generate answer."
    
    def process_query(self, query: str) -> Dict[str, Any]:

        """Process a single query through the complete RAG pipeline"""
        start_time = time.time()
        try:
            logger.info(f"Processing query: {query[:100]}...")
            # Step 1: Retrieve documents
            if self.config.use_multi_query:

                documents = self.multi_query_retrieval(query)
            else:
                documents = self.retrieve_documents(query)
            if not documents:
                return {
                    "query": query,
                    "answer": "No relevant documents found.",
                    "num_documents": 0,
                    "processing_time": time.time() - start_time
                }
            # Limit the number of documents before reranking
            num_docs_to_rerank = self.config.retriever.search_k
            documents_to_rerank = documents[:num_docs_to_rerank]
            logger.info(f"Reranking top {len(documents_to_rerank)} retrieved documents")
            # Step 2: Rerank documents
            top_documents = self.rerank_documents(query, documents_to_rerank)
            # Step 3: Combine documents into context
            context = self.prompt_manager.combine_documents(top_documents)
            # Step 4: Generate answer
            answer = self.generate_answer(query, context)
            processing_time = time.time() - start_time
            result = {
                "query": query,
                "answer": answer,
                "num_documents": len(documents),
                "num_top_documents": len(top_documents),
                "processing_time": processing_time
            }
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "num_documents": 0,
                "processing_time": time.time() - start_time
            }
    
    def process_queries_from_file(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        """Process multiple queries from a file"""
        results = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f.readlines() if line.strip()]
            
            logger.info(f"Processing {len(queries)} queries from {input_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, query in enumerate(queries, 1):
                    logger.info(f"Processing query {i}/{len(queries)}")
                    
                    result = self.process_query(query)
                    results.append(result)
                    
                    # Write result to file
                    f.write(f"Query: {result['query']}\n")
                    f.write(f"Answer: {result['answer']}\n")
                    f.write(f"Documents: {result['num_documents']}\n")
                    f.write(f"Time: {result['processing_time']:.2f}s\n")
                    f.write("=" * 50 + "\n")
                    f.flush()
                    
                    # Rate limiting
                    if i < len(queries):
                        time.sleep(self.config.request_delay)
            
            logger.info(f"Completed processing {len(queries)} queries. Results saved to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return results
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        health = {}
        
        try:
            health['llm_client'] = self.llm_client.health_check()
        except:
            health['llm_client'] = False
        
        try:
            health['embedding_manager'] = self.embedding_manager.vector_db is not None
        except:
            health['embedding_manager'] = False
        
        try:
            health['reranker'] = self.reranker.is_available() if self.reranker else True
        except:
            health['reranker'] = False
        
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            "config": {
                "use_reranking": self.config.use_reranking,
                "use_multi_query": self.config.use_multi_query,
                "use_few_shot": self.config.use_few_shot,
                "retriever_k": self.config.retriever.search_k,
                "top_k_after_rerank": self.config.retriever.top_k_after_rerank
            }
        }
        
        # Add database stats
        try:
            db_stats = self.embedding_manager.get_database_stats()
            stats["database"] = db_stats
        except:
            stats["database"] = {"error": "Unable to get database stats"}
        
        return stats