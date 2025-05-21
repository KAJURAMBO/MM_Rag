from typing import List, Dict, Union
import numpy as np
import faiss

class MultimodalRetriever:
    def __init__(self):
        """
        Initialize the multimodal retriever.
        """
        self.text_index = None
        self.image_index = None
        self.text_data = []
        self.image_data = []
        self.text_embedding_dim = None
        self.image_embedding_dim = None
    
    def build_indices(self, text_chunks: List[Dict], images: List[Dict]):
        """
        Build FAISS indices for both text and image embeddings.
        
        Args:
            text_chunks (List[Dict]): List of text chunks with embeddings
            images (List[Dict]): List of images with embeddings
        """
        # Build text index
        if text_chunks:
            text_embeddings = np.array([chunk['embedding'] for chunk in text_chunks])
            self.text_embedding_dim = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(self.text_embedding_dim)
            self.text_index.add(text_embeddings)
            self.text_data = text_chunks
        
        # Build image index
        if images:
            image_embeddings = np.array([img['embedding'] for img in images])
            self.image_embedding_dim = image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(self.image_embedding_dim)
            self.image_index.add(image_embeddings)
            self.image_data = images
    
    def search_text(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar text content using the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved text items with their metadata
        """
        results = []
        
        if self.text_index is not None and query_embedding.shape[0] == self.text_embedding_dim:
            text_distances, text_indices = self.text_index.search(query_embedding.reshape(1, -1), k)
            for dist, idx in zip(text_distances[0], text_indices[0]):
                if idx < len(self.text_data):
                    result = self.text_data[idx].copy()
                    result['score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                    results.append(result)
        
        return results
    
    def search_images(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar images using the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved image items with their metadata
        """
        results = []
        
        if self.image_index is not None and query_embedding.shape[0] == self.image_embedding_dim:
            image_distances, image_indices = self.image_index.search(query_embedding.reshape(1, -1), k)
            for dist, idx in zip(image_distances[0], image_indices[0]):
                if idx < len(self.image_data):
                    result = self.image_data[idx].copy()
                    result['score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                    results.append(result)
        
        return results 