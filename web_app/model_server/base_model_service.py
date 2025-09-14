#!/usr/bin/env python3
"""
Base Model Service - Uses LFM2-VL-450M for caption generation
Based on the instagram_caption_generator.py script
"""

import os
import sys
import torch
import base64
import io
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModelCaptionGenerator:
    def __init__(self, model_path="/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"):
        """Initialize the base model caption generator"""
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the LFM2-VL base model and processor"""
        try:
            logger.info("Loading LFM2-VL base model...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="bfloat16",
                trust_remote_code=True
            )
            
            logger.info("Base model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def generate_caption_from_base64(self, image_base64, style="instagram", max_tokens=128):
        """
        Generate caption from base64 encoded image
        
        Args:
            image_base64 (str): Base64 encoded image
            style (str): Caption style
            max_tokens (int): Maximum number of tokens to generate
        
        Returns:
            str: Generated caption
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Define style-specific prompts
            style_prompts = {
                "instagram": "Create an engaging Instagram caption for this image. Make it trendy, relatable, and use relevant hashtags. Keep it under 220 characters.",
                "professional": "Write a professional, descriptive caption for this image suitable for business or professional social media.",
                "casual": "Write a casual, friendly caption for this image that feels natural and conversational.",
                "creative": "Write a creative, artistic caption for this image that captures the mood and aesthetic."
            }
            
            prompt = style_prompts.get(style, style_prompts["instagram"])
            
            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            logger.info(f"Processing image with size: {image.size}, mode: {image.mode}")
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)
            
            # Generate caption
            logger.info("Generating base model caption...")
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Decode output
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up the caption (remove the prompt part)
            if "assistant" in caption:
                caption = caption.split("assistant")[-1].strip()
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating base model caption: {str(e)}")
            raise

# Global instance
base_model_generator = None

def get_base_model_generator():
    """Get or create the base model generator instance"""
    global base_model_generator
    if base_model_generator is None:
        base_model_generator = BaseModelCaptionGenerator()
    return base_model_generator
