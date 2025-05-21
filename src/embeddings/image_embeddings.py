from typing import List, Dict
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

class ImageEmbedder:
    def __init__(self):
        """
        Initialize the image embedder using ResNet50.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def generate_embeddings(self, images: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of images.
        
        Args:
            images (List[Dict]): List of images with metadata
            
        Returns:
            List[Dict]: List of images with their embeddings
        """
        for image_data in images:
            image = image_data['image']
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Add embedding to the image data
            image_data['embedding'] = embedding
        
        return images 