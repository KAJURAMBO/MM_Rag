# Import required libraries
from typing import List, Dict  # For type hints
import numpy as np  # For numerical operations and array handling
from sentence_transformers import SentenceTransformer  # For CLIP model
import logging  # For logging operations
from PIL import Image  # For image processing

class ImageEmbedder:
    """
    A class to handle image embedding generation using the CLIP model.
    CLIP (Contrastive Language-Image Pre-training) is a neural network trained to understand
    both images and text, making it ideal for multimodal applications.
    """
    
    def __init__(self):
        """
        Initialize the image embedder with CLIP model.
        Sets up logging and loads the CLIP model (clip-ViT-B-32) which is specifically
        designed for image-text understanding.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ImageEmbedder with CLIP model")
        
        # Initialize CLIP model
        # clip-ViT-B-32 is a Vision Transformer model with 32 layers
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.logger.info("CLIP model loaded successfully")

    def generate_embeddings(self, images: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of images using the CLIP model.
        
        The process involves:
        1. Converting images to the correct format (RGB)
        2. Converting PIL Images to numpy arrays
        3. Generating embeddings using CLIP
        4. Adding embeddings back to the image dictionaries
        
        Args:
            images (List[Dict]): List of image dictionaries containing PIL Image objects
                Each dictionary should have:
                - 'image': PIL Image object
                - 'page': Page number where the image was found
                - 'type': Type of content (should be 'image')
                - 'format': Image format (e.g., 'png', 'jpeg')
            
        Returns:
            List[Dict]: List of images with their embeddings added
                Each dictionary will now also contain:
                - 'embedding': numpy array of the image embedding
        """
        # Check if there are any images to process
        if not images:
            self.logger.warning("No images provided for embedding")
            return []
            
        self.logger.info(f"Generating embeddings for {len(images)} images")
        
        try:
            # Step 1: Convert PIL Images to numpy arrays and ensure correct format
            image_arrays = []
            for img in images:
                try:
                    # Get the PIL Image object from the dictionary
                    pil_image = img['image']
                    
                    # Ensure image is in RGB mode (required for CLIP)
                    # This handles cases where images might be in RGBA or other formats
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert PIL Image to numpy array for processing
                    img_array = np.array(pil_image)
                    image_arrays.append(img_array)
                    self.logger.info(f"Successfully converted image from page {img['page']} to numpy array")
                except Exception as e:
                    self.logger.error(f"Error converting image from page {img['page']} to numpy array: {str(e)}")
                    continue
            
            # Check if we have any valid images after conversion
            if not image_arrays:
                self.logger.error("No valid images to embed")
                return []
            
            # Step 2: Generate embeddings using CLIP model
            self.logger.info("Generating embeddings using CLIP model")
            # Convert numpy arrays back to PIL Images for CLIP model
            # CLIP expects PIL Images as input
            pil_images = [Image.fromarray(img_array) for img_array in image_arrays]
            # Generate embeddings and convert to numpy arrays
            embeddings = self.model.encode(pil_images, convert_to_numpy=True)
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 3: Add embeddings back to the original image dictionaries
            for img, embedding in zip(images, embeddings):
                img['embedding'] = embedding
                self.logger.info(f"Added embedding to image from page {img['page']}")
            
            return images
            
        except Exception as e:
            # Handle any unexpected errors during the embedding process
            self.logger.error(f"Error generating image embeddings: {str(e)}")
            return [] 