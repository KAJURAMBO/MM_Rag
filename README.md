# Multimodal RAG System

A Retrieval Augmented Generation (RAG) system that processes both text and images from PDF documents, enabling multimodal search and retrieval capabilities.

## System Architecture

The system is organized into several key components:

### 1. Document Processing (`src/document_processor/`)
- **PDFProcessor** (`pdf_processor.py`): Handles PDF document processing
  - Extracts text and images using PyMuPDF (primary) and PyPDF2 (fallback)
  - Supports multiple image formats (JPEG, PNG)
  - Handles image format conversion and preprocessing
  - Chunks text for better processing

### 2. Embeddings (`src/embeddings/`)
- **Text Embeddings** (`text_embeddings.py`):
  - Uses `all-MiniLM-L6-v2` model from Sentence Transformers
  - Generates 384-dimensional embeddings for text chunks
  - Optimized for semantic text search

- **Image Embeddings** (`image_embeddings.py`):
  - Uses `clip-ViT-B-32` model from Sentence Transformers
  - Generates 512-dimensional embeddings for images
  - Enables cross-modal search between text and images

### 3. Retrieval (`src/retrieval/`)
- **MultimodalRetriever** (`retriever.py`):
  - Maintains separate FAISS indices for text and images
  - Handles dimension-specific searches
  - Converts distances to similarity scores
  - Supports configurable number of results (k)

## How It Works

### 1. Document Processing
1. PDF is loaded and processed using PyMuPDF
2. Text is extracted and chunked into manageable pieces
3. Images are extracted and converted to RGB format
4. Both text and images are stored with metadata (page numbers, types)

### 2. Embedding Generation
1. Text chunks are embedded using MiniLM model
   - 384-dimensional vectors
   - Optimized for semantic similarity
2. Images are embedded using CLIP model
   - 512-dimensional vectors
   - Enables cross-modal understanding

### 3. Index Building
1. Separate FAISS indices are created for text and images
2. Text index uses 384-dimensional vectors
3. Image index uses 512-dimensional vectors
4. Both indices use L2 distance for similarity search

### 4. Search and Retrieval
1. Query processing:
   - Text queries use MiniLM model
   - Image queries use CLIP model
2. Separate searches for text and images
3. Results are ranked by similarity scores
4. Both text and image results are returned with metadata

## Models Used

1. **Text Processing**:
   - `all-MiniLM-L6-v2`: A lightweight but powerful model for text embeddings
   - Dimensions: 384
   - Use case: Semantic text search and similarity

2. **Image Processing**:
   - `clip-ViT-B-32`: A vision-language model for image understanding
   - Dimensions: 512
   - Use case: Cross-modal search and image-text matching

## Dependencies

- PyMuPDF: PDF processing and image extraction
- PyPDF2: Fallback PDF processing
- Sentence Transformers: Text and image embeddings
- FAISS: Efficient similarity search
- PIL: Image processing
- NumPy: Numerical operations
- Matplotlib: Image display

## Usage

1. Place your PDF file in the project directory as `input.pdf`
2. Run the main script:
   ```bash
   python main.py
   ```
3. The system will:
   - Process the PDF
   - Generate embeddings
   - Build search indices
   - Perform example search
   - Display results

## Current Limitations

1. **Image Search**:
   - Currently optimized for text search
   - Image search may require more specific queries
   - Cross-modal search (text-to-image) is experimental

2. **PDF Processing**:
   - Some complex PDF layouts may not be processed correctly
   - Image extraction may fail for certain PDF formats
   - Text chunking may split content in suboptimal ways

3. **Performance**:
   - Initial processing can be slow for large PDFs
   - Memory usage increases with document size
   - No caching of embeddings implemented yet

## Troubleshooting

### Common Issues

1. **No Image Results**:
   - Check if images were successfully extracted (check logs)
   - Verify image format compatibility
   - Try more specific image-related queries

2. **Text Search Issues**:
   - Adjust chunk size if text is split incorrectly
   - Try different query formulations
   - Check if text was properly extracted from PDF

3. **Performance Problems**:
   - Reduce PDF size if possible
   - Process smaller chunks of text
   - Consider using GPU if available

### Debugging Tips

1. Check the logs for:
   - Successful image extraction
   - Embedding generation
   - Search process details

2. Verify PDF content:
   - Ensure images are in supported formats
   - Check text extraction quality
   - Validate PDF structure

3. Query optimization:
   - Use specific, descriptive queries
   - Try different query formulations
   - Consider the context of your search

## Future Improvements

1. Add support for more document formats
2. Implement caching for embeddings
3. Add support for more embedding models
4. Implement hybrid search strategies
5. Add support for batch processing
6. Implement result reranking
7. Add support for custom queries
8. Improve image search capabilities
9. Add support for GPU acceleration
10. Implement better error handling and recovery
