from retriever import DocumentRetriever
from augmenter import ContextAugmenter
from generator import AnswerGenerator
import argparse
from config import Config

class VietnameseQASystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = Config()
        self.retriever = DocumentRetriever()
        self.augmenter = ContextAugmenter()
        self.generator = AnswerGenerator()
        self._initialized = True
    
    def answer(self, question: str) -> dict:
        try:
            # Step 1: Retrieve relevant documents
            retrieved = self.retriever.retrieve(question)
            
            # Step 2: Prepare context
            context = self.augmenter.augment(retrieved["documents"])
            
            # Step 3: Generate answer
            answer = self.generator.generate(question, context)
            
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {"title": meta.get("title", "Unknown"), "source": meta.get("source", "Unknown")} 
                    for meta in retrieved["metadatas"]
                ],
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"System error: {str(e)}",
                "sources": [],
                "success": False
            }

def main():
    parser = argparse.ArgumentParser(description="Vietnamese QA System")
    parser.add_argument("question", type=str, help="Question to answer")
    args = parser.parse_args()
    
    # Initialize system (will only load models once)
    qa_system = VietnameseQASystem()
    
    # Get answer
    result = qa_system.answer(args.question)
    
    # Display results
    print("\nKết quả:")
    print(f"Câu hỏi: {result['question']}")
    print(f"Câu trả lời: {result['answer']}")
    
    if result["sources"]:
        print("\nNguồn tham khảo:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['title']} ({source['source']})")

if __name__ == "__main__":
    main()