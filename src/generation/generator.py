from typing import List, Dict
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import time

class MultimodalGenerator:
    """
    A class to generate coherent multimodal outputs combining text and images using LLM.
    """
    
    def __init__(self, model_name: str = "facebook/opt-125m"):
        """
        Initialize the generator with logging setup and LLM.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize LLM
            self.logger.info(f"Loading LLM model: {model_name}")
            self.logger.info(f"Current working directory: {os.getcwd()}")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"PyTorch version: {torch.__version__}")
            
            # Try loading with pipeline first
            try:
                self.logger.info("Attempting to load model with pipeline...")
                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.logger.info("Successfully loaded model with pipeline")
            except Exception as pipeline_error:
                self.logger.error(f"Pipeline loading failed: {str(pipeline_error)}")
                self.logger.info("Falling back to manual model loading...")
                
                # Fallback to manual loading
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.generator = None
                self.logger.info("Successfully loaded model manually")
            
        except Exception as e:
            self.logger.error(f"Error loading LLM model: {str(e)}")
            self.generator = None
            self.model = None
            self.tokenizer = None
    
    def _prepare_prompt(self, query: str, text_results: List[Dict], image_results: List[Dict]) -> str:
        """
        Prepare the prompt for the LLM by combining query and retrieved content.
        
        Args:
            query (str): The original query
            text_results (List[Dict]): Retrieved text results
            image_results (List[Dict]): Retrieved image results
            
        Returns:
            str: Formatted prompt for the LLM
        """
        # Sort results by page number
        text_results = sorted(text_results, key=lambda x: x['page'])
        image_results = sorted(image_results, key=lambda x: x['page'])
        
        # Prepare context from text results
        context = "Retrieved text content:\n"
        for result in text_results:
            context += f"\nPage {result['page']}:\n{result['text']}\n"
        
        # Add image information
        if image_results:
            context += "\nRetrieved images:\n"
            for img in image_results:
                context += f"\nPage {img['page']}: [Image with similarity score {img['score']:.4f}]\n"
        
        # Create the final prompt
        prompt = f"""Based on the following context, please answer the question. Include references to both text and images where relevant.

Context:
{context}

Question: {query}

Answer:"""
        
        self.logger.info(f"Prepared prompt: {prompt[:200]}...")  # Log first 200 chars of prompt
        return prompt
    
    def generate_response(self, query: str, text_results: List[Dict], image_results: List[Dict]) -> Dict:
        """
        Generate a coherent response using LLM that combines text and images.
        
        Args:
            query (str): The original query
            text_results (List[Dict]): List of text results with their metadata
            image_results (List[Dict]): List of image results with their metadata
            
        Returns:
            Dict: A dictionary containing the generated response with text and images
        """
        try:
            if self.generator is None and self.model is None:
                self.logger.error("LLM model not initialized")
                return {'answer': "Error: LLM model not available", 'sections': [], 'images': []}
            
            # Prepare the prompt
            prompt = self._prepare_prompt(query, text_results, image_results)
            self.logger.info("Generated prompt for LLM")
            
            # Generate response using LLM
            self.logger.info("Generating response with LLM...")
            
            if self.generator is not None:
                # Use pipeline
                try:
                    self.logger.info("Using pipeline for generation...")
                    outputs = self.generator(
                        prompt,
                        max_length=512,
                        min_length=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.generator.tokenizer.eos_token_id,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                    response_text = outputs[0]['generated_text'].split("Answer:")[-1].strip()
                    self.logger.info("Pipeline generation successful")
                except Exception as e:
                    self.logger.error(f"Error in pipeline generation: {str(e)}")
                    response_text = "I apologize, but I encountered an error while generating the response."
            else:
                # Use manual generation
                try:
                    self.logger.info("Using manual generation...")
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_length=512,
                        min_length=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                    response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response_text = response_text.split("Answer:")[-1].strip()
                    self.logger.info("Manual generation successful")
                except Exception as e:
                    self.logger.error(f"Error in manual generation: {str(e)}")
                    response_text = "I apologize, but I encountered an error while generating the response."
            
            self.logger.info(f"Generated response: {response_text[:200]}...")  # Log first 200 chars of response
            
            # Organize the response
            response = {
                'answer': response_text,
                'sections': [],
                'images': []
            }
            
            # Add sections with text and images
            current_page = 0
            current_section = {
                'text': '',
                'page': 0,
                'images': []
            }
            
            # Process text results
            for text_result in text_results:
                if text_result['page'] != current_page:
                    if current_section['text']:
                        response['sections'].append(current_section)
                    current_section = {
                        'text': text_result['text'],
                        'page': text_result['page'],
                        'images': []
                    }
                    current_page = text_result['page']
                else:
                    current_section['text'] += '\n' + text_result['text']
                
                # Add related images
                page_images = [img for img in image_results if img['page'] == current_page]
                for img in page_images:
                    current_section['images'].append({
                        'image': img['image'],
                        'score': img['score']
                    })
            
            # Add the last section
            if current_section['text']:
                response['sections'].append(current_section)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {'answer': f"Error generating response: {str(e)}", 'sections': [], 'images': []}
    
    def display_response(self, response: Dict):
        """
        Display the generated response with text and images.
        
        Args:
            response (Dict): The generated response containing sections and images
        """
        try:
            # Display the LLM's answer
            print("\nGenerated Answer:")
            print("="*80)
            print(response['answer'])
            print("="*80)
            
            # Display supporting content
            print("\nSupporting Content:")
            for section in response['sections']:
                print(f"\nPage {section['page']}:")
                print(section['text'])
                
                if section['images']:
                    print("\nRelated Images:")
                    for img_data in section['images']:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img_data['image'])
                        plt.axis('off')
                        plt.title(f"Similarity Score: {img_data['score']:.4f}")
                        plt.show()
                
                print("\n" + "="*80 + "\n")
                
        except Exception as e:
            self.logger.error(f"Error displaying response: {str(e)}")
    
    def format_response(self, response: Dict) -> str:
        """
        Format the response as a markdown string.
        
        Args:
            response (Dict): The generated response
            
        Returns:
            str: Formatted markdown string
        """
        try:
            # Create images directory if it doesn't exist
            images_dir = os.path.join("response_md", "generated_images")
            os.makedirs(images_dir, exist_ok=True)
            
            markdown = "# Generated Answer\n\n"
            markdown += response['answer'] + "\n\n"
            
            markdown += "# Supporting Content\n\n"
            for section in response['sections']:
                markdown += f"## Page {section['page']}\n\n"
                markdown += f"{section['text']}\n\n"
                
                if section['images']:
                    markdown += "### Related Images\n\n"
                    for i, img_data in enumerate(section['images'], 1):
                        try:
                            # Save the image
                            image_filename = f"image_{section['page']}_{i}.png"
                            image_path = os.path.join(images_dir, image_filename)
                            
                            # Convert PIL Image to RGB if needed
                            if img_data['image'].mode != 'RGB':
                                img_data['image'] = img_data['image'].convert('RGB')
                            
                            # Save the image
                            img_data['image'].save(image_path, format='PNG')
                            
                            # Use relative path in markdown
                            relative_path = os.path.join("generated_images", image_filename)
                            markdown += f"![Image {i}]({relative_path})  \n"
                            markdown += f"*Similarity Score: {img_data['score']:.4f}*\n\n"
                        except Exception as e:
                            self.logger.error(f"Error saving image: {str(e)}")
                            markdown += f"*Error saving image: {str(e)}*\n\n"
                
                markdown += "---\n\n"
            
            return markdown
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return ""