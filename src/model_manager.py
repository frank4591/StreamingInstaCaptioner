"""
Model Manager for Streaming Instagram Captioner

Manages LFM2-VL model loading and provides efficient inference for real-time streaming.
"""

import torch
import logging
import time
from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

class StreamingModelManager:
    """
    Manages LFM2-VL model for real-time streaming applications.
    
    Optimized for continuous inference with minimal latency.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_batch_size: int = 1,
        enable_warmup: bool = True
    ):
        """
        Initialize the streaming model manager.
        
        Args:
            model_path: Path to LFM2-VL model
            device: Device to run on (cuda/cpu)
            max_batch_size: Maximum batch size for inference
            enable_warmup: Whether to warm up the model
        """
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.enable_warmup = enable_warmup
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        logging.info(f"StreamingModelManager initialized for {model_path}")
    
    def load_model(self) -> bool:
        """Load the LFM2-VL model for streaming."""
        try:
            logging.info(f"Loading LFM2-VL model from {self.model_path}...")
            
            # Check if model path exists
            if not Path(self.model_path).exists():
                logging.error(f"Model path {self.model_path} does not exist")
                return False
            
            # Load model and processor
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.is_loaded = True
            logging.info(f"LFM2-VL model loaded successfully on {self.model.device}")
            
            # Warm up the model if enabled
            if self.enable_warmup:
                self._warmup_model()
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def _warmup_model(self):
        """Warm up the model with dummy data."""
        try:
            logging.info("Warming up model...")
            
            # Create dummy image and text
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dummy_text = "Describe this image."
            
            # Run a few warmup inferences
            for _ in range(3):
                self.generate_caption(
                    image=dummy_image,
                    text_prompt=dummy_text,
                    temperature=0.1
                )
            
            logging.info("Model warmup completed")
            
        except Exception as e:
            logging.warning(f"Model warmup failed: {str(e)}")
    
    def generate_caption(
        self,
        image: Union[str, np.ndarray],
        text_prompt: str = "Describe this image in detail.",
        temperature: float = 0.1,
        max_new_tokens: int = 100
    ) -> Dict[str, any]:
        """
        Generate caption for streaming inference.
        
        Args:
            image: Input image (path or numpy array)
            text_prompt: Text prompt for generation
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary containing caption and metadata
        """
        if not self.is_loaded:
            return {
                "caption": "Model not loaded",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": "Model not loaded"
            }
        
        start_time = time.time()
        
        try:
            # Preprocess image
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                from PIL import Image
                image = Image.fromarray(image)
            
            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)
            
            # Generate with streaming-optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    min_p=0.15,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract clean caption
            clean_caption = self._extract_clean_caption(caption)
            
            processing_time = time.time() - start_time
            
            # Track performance
            self.inference_times.append(processing_time)
            self.total_inferences += 1
            
            # Keep only recent inference times for rolling average
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return {
                "caption": clean_caption,
                "raw_output": caption,
                "confidence": self._calculate_confidence(outputs),
                "processing_time": processing_time,
                "prompt": text_prompt
            }
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            return {
                "caption": "Error generating caption",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _extract_clean_caption(self, raw_output: str) -> str:
        """Extract clean caption from raw model output."""
        if not raw_output:
            return "A beautiful moment captured in time."
        
        import re
        
        # Look for the last "assistant" response
        parts = re.split(r'assistant\s*', raw_output, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            last_response = parts[-1].strip()
            
            # Look for content in quotes
            quote_match = re.search(r'["\']([^"\']*?)["\']', last_response)
            if quote_match:
                return quote_match.group(1).strip()
            
            # If no quotes, take everything after the last assistant
            if last_response and len(last_response) > 10:
                return last_response
        
        # Fallback: clean the entire output
        return self._clean_text(raw_output)
    
    def _clean_text(self, text: str) -> str:
        """Clean up text output."""
        if not text:
            return "A beautiful moment captured in time."
        
        # Remove common artifacts
        artifacts = ['<|reserved_', '<|endoftext|>', '<|startoftext|>', '<pad>', '<unk>']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Clean whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if text else "A beautiful moment captured in time."
    
    def _calculate_confidence(self, outputs) -> float:
        """Calculate confidence score from model outputs."""
        try:
            # Simple confidence based on output length and token probabilities
            if hasattr(outputs, 'scores') and outputs.scores:
                # Use average log probability as confidence
                scores = torch.stack(outputs.scores)
                avg_log_prob = torch.mean(scores).item()
                confidence = min(max(avg_log_prob / 10.0, 0.0), 1.0)
                return confidence
            else:
                # Fallback confidence based on output length
                output_length = len(outputs[0]) if len(outputs) > 0 else 0
                return min(output_length / 100.0, 1.0)
        except:
            return 0.5  # Default moderate confidence
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                "total_inferences": 0,
                "avg_processing_time": 0.0,
                "min_processing_time": 0.0,
                "max_processing_time": 0.0
            }
        
        return {
            "total_inferences": self.total_inferences,
            "avg_processing_time": np.mean(self.inference_times),
            "min_processing_time": np.min(self.inference_times),
            "max_processing_time": np.max(self.inference_times),
            "recent_avg_time": np.mean(self.inference_times[-10:]) if len(self.inference_times) >= 10 else np.mean(self.inference_times)
        }
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.is_loaded and self.model is not None and self.processor is not None
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.is_loaded = False
        logging.info("Model unloaded")


