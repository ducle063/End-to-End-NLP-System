#!/usr/bin/env python3

import os
import argparse
import json
from pathlib import Path

from config import RAGConfig, GeminiConfig, RetrieverConfig
from rag_pipeline import RAGPipeline
from utils import setup_logging

def create_config(
    gemini_api_key: str = None,
    use_reranking: bool = True,
    use_multi_query: bool = True,
    use_few_shot: bool = False,
    search_k: int = 100,
    top_k: int = 10
) -> RAGConfig:
    """Create RAG configuration"""
    
    config = RAGConfig()
    
    # Gemini configuration
    if gemini_api_key:
        config.gemini.api_key = gemini_api_key
    
    # Pipeline settings
    config.use_reranking = use_reranking
    config.use_multi_query = use_multi_query
    config.use_few_shot = use_few_shot
    
    # Retriever settings
    config.retriever.search_k = search_k
    config.retriever.top_k_after_rerank = top_k
    
    return config

def run_single_query(pipeline: RAGPipeline, query: str):
    """Run a single query through the pipeline"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    result = pipeline.process_query(query)
    
    print(f"Answer: {result['answer']}")
    print(f"Documents retrieved: {result['num_documents']}")
    print(f"Top documents used: {result['num_top_documents']}")
    print(f"Processing time: {result['processing_time']:.2f}s")

def run_batch_processing(pipeline: RAGPipeline, input_file: str, output_file: str):
    """Run batch processing on questions from file"""
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    print(f"Processing queries from {input_file}...")
    results = pipeline.process_queries_from_file(input_file, output_file)
    
    if results:
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"\nBatch processing completed:")
        print(f"- Total queries: {len(results)}")
        print(f"- Average processing time: {avg_time:.2f}s")
        print(f"- Results saved to: {output_file}")

def run_pipeline_comparison(input_file: str):
    """Compare different pipeline configurations"""
    
    configs = {
        "baseline": create_config(use_reranking=False, use_multi_query=False),
        "with_reranking": create_config(use_reranking=True, use_multi_query=False),
        "with_multi_query": create_config(use_reranking=False, use_multi_query=True),
        "full_pipeline": create_config(use_reranking=True, use_multi_query=True),
    }
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Running pipeline: {config_name.upper()}")
        print(f"{'='*80}")
        
        try:
            pipeline = RAGPipeline(config)
            output_file = f"results_{config_name}.txt"
            run_batch_processing(pipeline, input_file, output_file)
        except Exception as e:
            print(f"Error running {config_name}: {e}")

def interactive_mode():
    """Run interactive question-answering mode"""
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
    
    # Create configuration
    config = create_config(gemini_api_key=api_key)
    
    try:
        # Initialize pipeline
        print("Initializing RAG Pipeline...")
        pipeline = RAGPipeline(config)
        
        # Health check
        health = pipeline.health_check()
        print(f"Health check: {health}")
        
        if not all(health.values()):
            print("Warning: Some components are not healthy")
        
        # Get stats
        stats = pipeline.get_stats()
        print(f"Pipeline stats: {json.dumps(stats, indent=2)}")
        
        print("\n" + "="*60)
        print("Interactive RAG System")
        print("Type 'quit' to exit, 'stats' for statistics")
        print("="*60)
        
        while True:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'stats':
                stats = pipeline.get_stats()
                print(json.dumps(stats, indent=2))
                continue
            elif not query:
                continue
            
            run_single_query(pipeline, query)
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Pipeline with Gemini")
    parser.add_argument("--mode", choices=["interactive", "batch", "compare"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--input", help="Input file for batch processing")
    parser.add_argument("--output", help="Output file for batch processing")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--no-reranking", action="store_true", 
                       help="Disable reranking")
    parser.add_argument("--no-multi-query", action="store_true", 
                       help="Disable multi-query retrieval")
    parser.add_argument("--few-shot", action="store_true", 
                       help="Enable few-shot prompting")
    parser.add_argument("--search-k", type=int, default=100, 
                       help="Number of documents to retrieve")
    parser.add_argument("--top-k", type=int, default=10, 
                       help="Number of top documents after reranking")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.mode == "interactive":
        if args.query:
            # Single query mode
            config = create_config(
                use_reranking=not args.no_reranking,
                use_multi_query=not args.no_multi_query,
                use_few_shot=args.few_shot,
                search_k=args.search_k,
                top_k=args.top_k
            )
            
            pipeline = RAGPipeline(config)
            run_single_query(pipeline, args.query)
        else:
            # Interactive mode
            interactive_mode()
    
    elif args.mode == "batch":
        if not args.input:
            print("Error: --input required for batch mode")
            return
        
        output_file = args.output or "rag_results.txt"
        
        config = create_config(
            use_reranking=not args.no_reranking,
            use_multi_query=not args.no_multi_query,
            use_few_shot=args.few_shot,
            search_k=args.search_k,
            top_k=args.top_k
        )
        
        pipeline = RAGPipeline(config)
        pipeline.request_delay = config.request_delay  # Set rate limiting
        run_batch_processing(pipeline, args.input, output_file)
    
    elif args.mode == "compare":
        if not args.input:
            print("Error: --input required for compare mode")
            return
        
        run_pipeline_comparison(args.input)

if __name__ == "__main__":
    main()