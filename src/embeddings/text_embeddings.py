from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text embedder with a specific model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts (List[Dict]): List of text chunks with metadata
            
        Returns:
            List[Dict]: List of text chunks with their embeddings
        """
        text_contents = [item['text'] for item in texts]
        embeddings = self.model.encode(text_contents)
        
        # Add embeddings to the original text chunks
        for i, text_chunk in enumerate(texts):
            text_chunk['embedding'] = embeddings[i]
        
        return texts 