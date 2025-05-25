# Quick check for reranker functionality
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def quick_reranker_check():
    print("🔍 Quick Reranker Check")
    
    # 1. Check if FlagEmbedding is installed
    try:
        from FlagEmbedding import FlagReranker
        print("✅ FlagEmbedding is installed")
    except ImportError:
        print("❌ FlagEmbedding not installed")
        print("Install with: pip install FlagEmbedding")
        return False
    
    # 2. Try to create a reranker instance
    try:
        reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        print("✅ Reranker model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load reranker model: {e}")
        return False
    
    # 3. Test basic functionality
    try:
        query = "What is Python?"
        documents = ["Python is a programming language", "Java is also a programming language"]
        
        # Test reranking
        pairs = [[query, doc] for doc in documents]
        scores = reranker.compute_score(pairs)
        
        print(f"✅ Reranker working! Scores: {scores}")
        return True
        
    except Exception as e:
        print(f"❌ Reranker test failed: {e}")
        return False

if __name__ == "__main__":
    quick_reranker_check()