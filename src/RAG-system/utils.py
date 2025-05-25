import logging
from typing import List, Sequence
from langchain.schema import Document

logger = logging.getLogger(__name__)

def unique_documents(documents: Sequence[Document]) -> List[Document]:
    """Remove duplicate documents from a sequence"""
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]

def split_questions(string: str, delimiter: str = '?') -> List[str]:
    """Split string into questions based on delimiter"""
    questions = string.split(delimiter)
    questions = [question.strip().split(". ", 1)[-1] + delimiter for question in questions if question.strip()]
    # Remove the last empty question if it exists
    if questions and not questions[-1].replace(delimiter, '').strip():
        questions = questions[:-1]
    return questions

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text.strip()

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a space in the last 20%
        truncated = truncated[:last_space]
    
    return truncated + "..."

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log')
        ]
    )

def validate_documents(documents: List[Document]) -> List[Document]:
    """Validate and clean documents"""
    valid_docs = []
    
    for doc in documents:
        if doc.page_content and doc.page_content.strip():
            # Clean the content
            doc.page_content = clean_text(doc.page_content)
            valid_docs.append(doc)
        else:
            logger.warning("Skipping document with empty content")
    
    return valid_docs

def format_answer(answer: str) -> str:
    """Format and clean the generated answer"""
    if not answer:
        return "No answer generated."
    
    # Clean the answer
    answer = clean_text(answer)
    
    # Remove common prefixes that models sometimes add
    prefixes_to_remove = [
        "Answer:",
        "Response:",
        "The answer is:",
        "Based on the context:",
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    return answer
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