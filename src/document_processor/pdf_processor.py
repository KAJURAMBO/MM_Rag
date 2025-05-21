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
        # Create debug_images directory if it doesn't exist
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

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
            self.logger.info(f"Opened PDF with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    self.logger.info(f"Processing page {page_num + 1}")
                    
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        text_chunks.append({
                            'text': text,
                            'page': page_num + 1,
                            'type': 'text'
                        })
                        self.logger.info(f"Extracted text from page {page_num + 1}")
                    
                    # Extract images with more detailed logging
                    image_list = page.get_images(full=True)
                    self.logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            self.logger.info(f"Extracted image {img_index + 1} from page {page_num + 1} with format {image_ext}")
                            
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
                                
                                # Save image temporarily for debugging
                                debug_path = os.path.join(self.debug_dir, f"debug_image_p{page_num + 1}_i{img_index + 1}.{image_ext}")
                                image.save(debug_path)
                                self.logger.info(f"Saved debug image to {debug_path}")
                                
                                images.append({
                                    'image': image,
                                    'page': page_num + 1,
                                    'type': 'image',
                                    'format': image_ext
                                })
                                self.logger.info(f"Successfully processed image {img_index + 1} from page {page_num + 1}")
                                
                            except Exception as e:
                                self.logger.error(f"Error processing image {img_index + 1} from page {page_num + 1}: {str(e)}")
                                continue
                                
                        except Exception as e:
                            self.logger.error(f"Error extracting image {img_index + 1} from page {page_num + 1}: {str(e)}")
                            continue
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with PyMuPDF: {str(e)}")
            # Fallback to PyPDF2 if PyMuPDF fails
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    self.logger.info(f"Falling back to PyPDF2, found {len(pdf_reader.pages)} pages")
                    
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
                            self.logger.info(f"Extracted text from page {page_num + 1} using PyPDF2")
                        
                        # Extract images
                        try:
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
                                            
                                            # Save image temporarily for debugging
                                            safe_obj_name = obj.replace('/', '_').replace('\\', '_')
                                            debug_path = os.path.join(self.debug_dir, f"debug_image_p{page_num + 1}_pypdf2_{safe_obj_name}.png")
                                            image.save(debug_path)
                                            self.logger.info(f"Saved debug image from PyPDF2 to {debug_path}")
                                            
                                            # Add image to results
                                            images.append({
                                                'image': image,
                                                'page': page_num + 1,
                                                'type': 'image',
                                                'format': 'png'
                                            })
                                            self.logger.info(f"Successfully extracted image from page {page_num + 1} using PyPDF2")
                                            
                                        except Exception as e:
                                            self.logger.error(f"Error extracting image from page {page_num + 1} using PyPDF2: {str(e)}")
                                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing XObjects on page {page_num + 1}: {str(e)}")
                            continue
            except Exception as e:
                self.logger.error(f"Error processing PDF with PyPDF2: {str(e)}")
                raise
        
        self.logger.info(f"Extraction complete. Found {len(text_chunks)} text chunks and {len(images)} images")
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