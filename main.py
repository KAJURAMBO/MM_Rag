import os
import logging
import traceback

# Set environment variables for HuggingFace cache
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), '.cache', 'huggingface')

from src.document_processor.pdf_processor import PDFProcessor
from src.embeddings.text_embeddings import TextEmbedder
from src.embeddings.image_embeddings import ImageEmbedder
from src.retrieval.retriever import MultimodalRetriever
from src.generation.generator import MultimodalGenerator
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

def process_query(query: str, retriever: MultimodalRetriever, generator: MultimodalGenerator):
    """
    Process a single query and display results.
    Shows only top 2 results based on similarity scores.
    
    Args:
        query (str): The search query
        retriever (MultimodalRetriever): The retriever instance
        generator (MultimodalGenerator): The generator instance
    """
    logger.info(f"Processing query: {query}")
    
    # Generate text query embedding
    text_query_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_query_embedding = text_query_model.encode(query)
    
    # Generate image query embedding using CLIP's text encoder
    image_query_model = SentenceTransformer('clip-ViT-B-32')
    # CLIP expects text queries to be wrapped in a list
    image_query_embedding = image_query_model.encode([query], convert_to_numpy=True)[0]
    
    # Search for text and images separately
    text_results = retriever.search_text(text_query_embedding, k=5)  # Get 5 results but display top 2
    image_results = retriever.search_images(image_query_embedding, k=5)  # Get 5 results but display top 2
    
    # Sort results by similarity score and take top 2
    text_results = sorted(text_results, key=lambda x: x['score'], reverse=True)[:2]
    image_results = sorted(image_results, key=lambda x: x['score'], reverse=True)[:2]
    
    # Generate coherent response using the generator
    response = generator.generate_response(query, text_results, image_results)
    
    # Display text results
    if text_results:
        logger.info(f"Found {len(text_results)} text results")
        print("\nTop 2 Text Results:")
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
        print("\nTop 2 Image Results:")
        for i, result in enumerate(image_results, 1):
            print(f"\nImage Result {i}:")
            print(f"Image found on page {result['page']}")
            print(f"Similarity Score: {result['score']:.4f}")
            try:
                # Display the actual image from the result, not the debug image
                display_image(result['image'])
            except Exception as e:
                logger.error(f"Error displaying image: {str(e)}")
    else:
        logger.info("No image results found")
        print("\nNo image results found.")
    
    # Display generated response
    print("\nGenerated Response:")
    generator.display_response(response)
    
    # Ask if user wants to save results
    save = input("\nWould you like to save the results? (y/n): ").strip().lower()
    if save == 'y':
        try:
            # Create response_md directory if it doesn't exist
            os.makedirs("response_md", exist_ok=True)
            
            # Clean the query string to make it safe for filenames
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_query = safe_query.replace(' ', '_')
            
            # Ensure we're saving in response_md folder
            output_path = os.path.join("response_md", f"query_results_{safe_query}.md")
            
            # Generate markdown content
            markdown = generator.format_response(response)
            
            # Save the markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")

def main():
    try:
        # Create necessary directories
        os.makedirs("response_md", exist_ok=True)
        
        # Initialize components
        pdf_processor = PDFProcessor()
        text_embedder = TextEmbedder()
        image_embedder = ImageEmbedder()
        retriever = MultimodalRetriever()
        
        # Initialize generator at startup
        logger.info("Initializing generator...")
        generator = MultimodalGenerator()
        logger.info("Generator initialized successfully")
        
        # Process PDF
        pdf_path = "input_2.pdf"  # Replace with your PDF path
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file '{pdf_path}' not found.")
            return
            
        logger.info(f"Processing PDF: {pdf_path}")
        text_chunks, images = pdf_processor.extract_text_and_images(pdf_path)
        
        # Add debug logging for images
        logger.info(f"Number of images extracted: {len(images)}")
        for i, img in enumerate(images):
            logger.info(f"Image {i+1}: Page {img['page']}, Format: {img['format']}")
        
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
            # Add debug logging for image embeddings
            logger.info(f"Number of images with embeddings: {len(images_with_embeddings)}")
            for i, img in enumerate(images_with_embeddings):
                logger.info(f"Image {i+1} embedding shape: {img['embedding'].shape}")
        else:
            images_with_embeddings = []
        
        # Build search indices
        logger.info("Building search indices")
        retriever.build_indices(text_chunks_with_embeddings, images_with_embeddings)
        
        # Interactive query loop
        print("\nWelcome to the Multimodal RAG System!")
        print("You can now search through the PDF document.")
        print("Type 'exit' or 'quit' to end the session.")
        print("\nExample queries:")
        print("- 'Show me the image on page 5'")
        print("- 'Find text about marketing strategy'")
        print("- 'Show me diagrams related to sales'")
        
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("Thank you for using the Multimodal RAG System!")
                break
                
            if not query:
                print("Please enter a valid query.")
                continue
                
            process_query(query, retriever, generator)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
