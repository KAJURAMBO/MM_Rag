import os
import logging
import traceback
from src.document_processor.pdf_processor import PDFProcessor
from src.embeddings.text_embeddings import TextEmbedder
from src.embeddings.image_embeddings import ImageEmbedder
from src.retrieval.retriever import MultimodalRetriever
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_image(image_data):
    """
    Display an image using matplotlib.
    
    Args:
        image_data: PIL Image object
    """
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data)
        plt.axis('off')
        plt.show()
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")

def main():
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        text_embedder = TextEmbedder()
        image_embedder = ImageEmbedder()
        retriever = MultimodalRetriever()
        
        # Process PDF
        pdf_path = "input.pdf"  # Replace with your PDF path
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file '{pdf_path}' not found.")
            return
            
        logger.info(f"Processing PDF: {pdf_path}")
        text_chunks, images = pdf_processor.extract_text_and_images(pdf_path)
        
        if not text_chunks and not images:
            logger.warning("No text or images found in the PDF.")
            return
            
        # Generate embeddings
        if text_chunks:
            logger.info(f"Generating embeddings for {len(text_chunks)} text chunks")
            text_chunks_with_embeddings = text_embedder.generate_embeddings(text_chunks)
        else:
            text_chunks_with_embeddings = []
            
        if images:
            logger.info(f"Generating embeddings for {len(images)} images")
            images_with_embeddings = image_embedder.generate_embeddings(images)
        else:
            images_with_embeddings = []
        
        # Build search indices
        logger.info("Building search indices")
        retriever.build_indices(text_chunks_with_embeddings, images_with_embeddings)
        
        # Example query
        query = "Show me the image on fifth page"
        logger.info(f"Processing query: {query}")
        
        # Generate text query embedding
        text_query_model = SentenceTransformer('all-MiniLM-L6-v2')
        text_query_embedding = text_query_model.encode(query)
        
        # Generate image query embedding
        image_query_model = SentenceTransformer('clip-ViT-B-32')
        image_query_embedding = image_query_model.encode(query)
        
        # Search for text and images separately
        text_results = retriever.search_text(text_query_embedding, k=5)
        image_results = retriever.search_images(image_query_embedding, k=5)
        
        # Display text results
        if text_results:
            logger.info(f"Found {len(text_results)} text results")
            print("\nText Results:")
            for i, result in enumerate(text_results, 1):
                print(f"\nText Result {i}:")
                print(f"Text: {result['text']}")
                print(f"Page: {result['page']}")
                print(f"Similarity Score: {result['score']:.4f}")
        else:
            logger.info("No text results found")
            print("\nNo text results found.")
        
        # Display image results
        if image_results:
            logger.info(f"Found {len(image_results)} image results")
            print("\nImage Results:")
            for i, result in enumerate(image_results, 1):
                print(f"\nImage Result {i}:")
                print(f"Image found on page {result['page']}")
                print(f"Similarity Score: {result['score']:.4f}")
                try:
                    display_image(result['image'])
                except Exception as e:
                    logger.error(f"Error displaying image: {str(e)}")
        else:
            logger.info("No image results found")
            print("\nNo image results found.")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
