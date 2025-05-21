import os
from typing import List, Tuple, Dict
import PyPDF2
from PIL import Image
import io
import logging
import fitz  # PyMuPDF

class PDFProcessor:
    def __init__(self):
        self.supported_image_types = ['.jpg', '.jpeg', '.png']
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_text_and_images(self, pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract text and images from a PDF file using PyMuPDF for better image extraction.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Tuple[List[Dict], List[Dict]]: List of text chunks and list of image data
        """
        text_chunks = []
        images = []
        
        try:
            # First try with PyMuPDF for better image extraction
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_chunks.append({
                        'text': text,
                        'page': page_num + 1,
                        'type': 'text'
                    })
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Try to open the image
                        try:
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Convert to RGB if necessary
                            if image.mode in ['RGBA', 'LA']:
                                background = Image.new('RGB', image.size, (255, 255, 255))
                                background.paste(image, mask=image.split()[-1])
                                image = background
                            elif image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            images.append({
                                'image': image,
                                'page': page_num + 1,
                                'type': 'image'
                            })
                            self.logger.info(f"Successfully extracted image {img_index + 1} from page {page_num + 1}")
                            
                        except Exception as e:
                            self.logger.error(f"Error processing image {img_index + 1} from page {page_num + 1}: {str(e)}")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"Error extracting image {img_index + 1} from page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with PyMuPDF: {str(e)}")
            # Fallback to PyPDF2 if PyMuPDF fails
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        
                        # Extract text
                        text = page.extract_text()
                        if text.strip():
                            text_chunks.append({
                                'text': text,
                                'page': page_num + 1,
                                'type': 'text'
                            })
                        
                        # Extract images
                        if '/Resources' in page and '/XObject' in page['/Resources']:
                            x_objects = page['/Resources']['/XObject'].get_object()
                            
                            for obj in x_objects:
                                if x_objects[obj]['/Subtype'] == '/Image':
                                    try:
                                        image_data = x_objects[obj].get_data()
                                        image = Image.open(io.BytesIO(image_data))
                                        
                                        # Convert to RGB if necessary
                                        if image.mode in ['RGBA', 'LA']:
                                            background = Image.new('RGB', image.size, (255, 255, 255))
                                            background.paste(image, mask=image.split()[-1])
                                            image = background
                                        elif image.mode != 'RGB':
                                            image = image.convert('RGB')
                                        
                                        images.append({
                                            'image': image,
                                            'page': page_num + 1,
                                            'type': 'image'
                                        })
                                        self.logger.info(f"Successfully extracted image from page {page_num + 1} using PyPDF2")
                                        
                                    except Exception as e:
                                        self.logger.error(f"Error extracting image from page {page_num + 1} using PyPDF2: {str(e)}")
                                        continue
            except Exception as e:
                self.logger.error(f"Error processing PDF with PyPDF2: {str(e)}")
                raise
        
        return text_chunks, images

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into smaller chunks for processing.
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks 