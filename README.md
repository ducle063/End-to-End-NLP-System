# End-to-End-NLP-System

# RAG Pipeline with Google Gemini

A modular Retrieval-Augmented Generation (RAG) system using Google Gemini API for answer generation, with support for document reranking and multi-query retrieval.

## Features

- **Modular Architecture**: Separate components for easy maintenance and testing
- **Google Gemini Integration**: Uses Gemini API for high-quality answer generation
- **Advanced Retrieval**: Multi-query retrieval and document reranking
- **Flexible Configuration**: Easy to customize pipeline behavior
- **Multiple Run Modes**: Interactive, batch processing, and comparison modes

## Components

### Core Components

1. **config.py** - Configuration management
2. **llm_client.py** - Gemini API client
3. **embedding_manager.py** - Vector embeddings and database
4. **reranker.py** - Document reranking using BGE reranker
5. **prompt_manager.py** - Prompt templates and formatting
6. **utils.py** - Utility functions
7. **rag_pipeline.py** - Main pipeline orchestration
8. **main.py** - CLI interface and usage examples

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your Google Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

3. Ensure you have a ChromaDB vector database in the `db/` directory with your documents.

## Usage

### Interactive Mode

```bash
python main.py --mode interactive
```

This will start an interactive session where you can ask questions.

### Single Query

```bash
python main.py --mode interactive --query "What is Graham Neubig's title?"
```

### Batch Processing

```bash
python main.py --mode batch --input questions.txt --output results.txt
```

### Pipeline Comparison

Compare different pipeline configurations:

```bash
python main.py --mode compare --input questions.txt
```

### Configuration Options

```bash
# Disable reranking
python main.py --no-reranking

# Disable multi-query retrieval
python main.py --no-multi-query

# Enable few-shot prompting
python main.py --few-shot

# Adjust retrieval parameters
python main.py --search-k 50 --top-k 5
```

## Configuration

The system uses a hierarchical configuration system. Key settings:

```python
from config import RAGConfig

config = RAGConfig()
config.use_reranking = True          # Enable document reranking
config.use_multi_query = True        # Enable multi-query retrieval
config.use_few_shot = False          # Enable few-shot prompting
config.retriever.search_k = 100      # Documents to retrieve
config.retriever.top_k_after_rerank = 3  # Top documents for generation
```

## Pipeline Modes

### 1. Basic RAG
- Simple retrieval + generation
- Fast but may miss relevant documents

### 2. RAG + Reranking
- Retrieves more documents, then reranks by relevance
- Better accuracy, slightly slower

### 3. RAG + Multi-Query
- Generates paraphrased queries for better recall
- Retrieves documents for all query variations

### 4. Full Pipeline
- Combines reranking + multi-query + optional few-shot
- Best accuracy, highest latency

## API Usage

```python
from config import RAGConfig
from rag_pipeline import RAGPipeline

# Create configuration
config = RAGConfig()
config.gemini.api_key = "your_api_key"

# Initialize pipeline
pipeline = RAGPipeline(config)

# Process single query
result = pipeline.process_query("What is Carnegie Mellon's ranking?")
print(result['answer'])

# Health check
health = pipeline.health_check()
print(health)
```

## File Structure

```
rag-pipeline/
├── config.py                 # Configuration classes
├── llm_client.py             # Gemini API client
├── embedding_manager.py      # Embeddings and vector DB
├── reranker.py              # Document reranking
├── prompt_manager.py        # Prompt templates
├── utils.py                 # Utility functions
├── rag_pipeline.py          # Main pipeline
├── main.py                  # CLI interface
├── requirements.txt         # Dependencies
├── README.md               # This file
├── db/                     # ChromaDB vector database
├── questions.txt           # Input questions
└── results/                # Output results
    ├── rag_baseline.txt
    ├── rag_with_reranking.txt
    └── rag_full_pipeline.txt
```

## Performance Comparison

Based on the original implementation, performance typically improves as:

1. **Baseline RAG**: Fast, moderate accuracy
2. **+ Reranking**: +15-20% accuracy, +50% latency
3. **+ Multi-Query**: +10-15% recall, +100% latency
4. **+ Few-Shot**: Variable results (as noted in original code)

## Error Handling

The system includes comprehensive error handling:

- API rate limiting and retry logic
- Graceful degradation when components fail
- Detailed logging for debugging
- Health checks for all components

## Logging

Logs are written to both console and `rag_system.log`:

```python
import logging
logging.getLogger('rag_pipeline').setLevel(logging.INFO)
```

## Troubleshooting

### Common Issues

1. **Gemini API Key Issues**
   - Ensure `GEMINI_API_KEY` is set
   - Check API key permissions

2. **Vector Database Issues**
   - Ensure `db/` directory exists with embeddings
   - Check ChromaDB version compatibility

3. **Reranker Issues**
   - FlagEmbedding requires specific PyTorch versions
   - Disable reranking if having issues: `--no-reranking`

4. **Memory Issues**
   - Reduce `search_k` parameter
   - Use CPU instead of GPU for embeddings

### Performance Tips

1. **For faster processing**:
   - Disable reranking and multi-query
   - Reduce `search_k` to 20-50

2. **For better accuracy**:
   - Enable all features
   - Increase `search_k` to 200+
   - Tune prompt templates

## Contributing

1. Follow the modular architecture
2. Add comprehensive error handling
3. Include logging for debugging
4. Write tests for new components
5. Update documentation

## License

This project is provided as-is for educational and research purposes.