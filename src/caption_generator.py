"""
Caption Generator for Streaming Instagram Captioner

Handles real-time Instagram caption generation with video context.
"""

import logging
import time
from typing import Dict, Optional, Callable, List
import numpy as np

class StreamingCaptionGenerator:
    """
    Generates Instagram captions in real-time using video context.
    
    Manages caption generation, style variations, and output formatting
    for streaming applications.
    """
    
    def __init__(
        self,
        model_manager,
        context_buffer,
        caption_style: str = "instagram",
        update_interval: float = 3.0,
        caption_callback: Optional[Callable] = None
    ):
        """
        Initialize the caption generator.
        
        Args:
            model_manager: StreamingModelManager instance
            context_buffer: StreamingContextBuffer instance
            caption_style: Style of captions to generate
            update_interval: How often to generate new captions (seconds)
            caption_callback: Callback function for new captions
        """
        self.model_manager = model_manager
        self.context_buffer = context_buffer
        self.caption_style = caption_style
        self.update_interval = update_interval
        self.caption_callback = caption_callback
        
        # Caption state
        self.current_caption = ""
        self.last_caption_time = 0.0
        self.caption_history = []
        self.max_history = 50
        
        # Style configurations
        self.style_configs = {
            "instagram": {
                "prompt_template": "Create an Instagram-style caption for this image with hashtags and emojis.",
                "max_length": 200,
                "include_hashtags": True,
                "include_emojis": True
            },
            "descriptive": {
                "prompt_template": "Describe this image in detail for accessibility purposes.",
                "max_length": 300,
                "include_hashtags": False,
                "include_emojis": False
            },
            "minimal": {
                "prompt_template": "Create a short, clean caption for this image.",
                "max_length": 100,
                "include_hashtags": False,
                "include_emojis": False
            },
            "trendy": {
                "prompt_template": "Create a trendy, modern caption for this image with popular hashtags.",
                "max_length": 250,
                "include_hashtags": True,
                "include_emojis": True
            }
        }
        
        # Performance tracking
        self.generation_times = []
        self.total_generations = 0
        
        logging.info(f"StreamingCaptionGenerator initialized: {caption_style} style, {update_interval}s interval")
    
    def generate_caption(
        self,
        frame: np.ndarray,
        force_update: bool = False
    ) -> Optional[Dict[str, any]]:
        """
        Generate a caption for the current frame.
        
        Args:
            frame: Current video frame
            force_update: Force generation even if interval hasn't passed
            
        Returns:
            Caption result dictionary or None if not generated
        """
        current_time = time.time()
        
        # Check if we should generate a new caption
        if not force_update and not self._should_generate_caption(current_time):
            return None
        
        # Check if we have enough context
        if not self.context_buffer.is_ready():
            logging.debug("Insufficient context for caption generation")
            return None
        
        # Get current context
        context_info = self.context_buffer.get_current_context()
        context_text = context_info["context_text"]
        
        if not context_text or context_text == "No context available":
            logging.debug("No context available for caption generation")
            return None
        
        # Generate caption
        start_time = time.time()
        
        try:
            # Create context-aware prompt
            prompt = self._create_context_prompt(context_text)
            
            # Generate caption using model
            result = self.model_manager.generate_caption(
                image=frame,
                text_prompt=prompt,
                temperature=0.1,
                max_new_tokens=150
            )
            
            if result.get("error"):
                logging.warning(f"Caption generation error: {result['error']}")
                return None
            
            # Process and format caption
            processed_caption = self._process_caption(result["caption"])
            
            # Create caption result
            caption_result = {
                "caption": processed_caption,
                "raw_caption": result["caption"],
                "context": context_text,
                "prompt": prompt,
                "style": self.caption_style,
                "confidence": result["confidence"],
                "processing_time": result["processing_time"],
                "generation_time": time.time() - start_time,
                "timestamp": current_time,
                "context_consistency": context_info["consistency"]
            }
            
            # Update state
            self.current_caption = processed_caption
            self.last_caption_time = current_time
            self._add_to_history(caption_result)
            
            # Track performance
            self.generation_times.append(caption_result["generation_time"])
            self.total_generations += 1
            
            if len(self.generation_times) > 100:
                self.generation_times = self.generation_times[-100:]
            
            # Call callback if provided
            if self.caption_callback:
                try:
                    self.caption_callback(caption_result)
                except Exception as e:
                    logging.warning(f"Caption callback error: {str(e)}")
            
            logging.debug(f"Generated caption: {processed_caption[:50]}...")
            return caption_result
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            return None
    
    def _should_generate_caption(self, current_time: float) -> bool:
        """Check if a new caption should be generated."""
        time_since_last = current_time - self.last_caption_time
        return time_since_last >= self.update_interval
    
    def _create_context_prompt(self, context_text: str) -> str:
        """Create a context-aware prompt for caption generation."""
        style_config = self.style_configs.get(self.caption_style, self.style_configs["instagram"])
        
        base_prompt = style_config["prompt_template"]
        
        # Add context information
        context_prompt = f"Based on the video context: '{context_text}', {base_prompt}"
        
        return context_prompt
    
    def _process_caption(self, raw_caption: str) -> str:
        """Process and format the generated caption."""
        if not raw_caption:
            return "A beautiful moment captured in time."
        
        # Clean up the caption
        caption = self._clean_caption(raw_caption)
        
        # Apply style-specific formatting
        style_config = self.style_configs.get(self.caption_style, self.style_configs["instagram"])
        
        # Truncate if too long
        max_length = style_config["max_length"]
        if len(caption) > max_length:
            caption = caption[:max_length-3] + "..."
        
        # Add hashtags if enabled
        if style_config["include_hashtags"] and self.caption_style in ["instagram", "trendy"]:
            caption = self._add_hashtags(caption)
        
        return caption
    
    def _clean_caption(self, caption: str) -> str:
        """Clean up the caption text."""
        if not caption:
            return "A beautiful moment captured in time."
        
        # Remove common artifacts
        artifacts = ['<|reserved_', '<|endoftext|>', '<|startoftext|>', '<pad>', '<unk>']
        for artifact in artifacts:
            caption = caption.replace(artifact, '')
        
        # Clean whitespace
        import re
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Remove quotes if present
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        elif caption.startswith("'") and caption.endswith("'"):
            caption = caption[1:-1]
        
        return caption if caption else "A beautiful moment captured in time."
    
    def _add_hashtags(self, caption: str) -> str:
        """Add relevant hashtags to the caption."""
        # Simple hashtag generation based on content
        hashtags = []
        
        # Common hashtags based on content analysis
        content_lower = caption.lower()
        
        if any(word in content_lower for word in ['sunset', 'sunrise', 'golden hour']):
            hashtags.extend(['#sunset', '#goldenhour', '#nature'])
        
        if any(word in content_lower for word in ['beach', 'ocean', 'water', 'wave']):
            hashtags.extend(['#beach', '#ocean', '#water', '#coastal'])
        
        if any(word in content_lower for word in ['mountain', 'hill', 'peak', 'summit']):
            hashtags.extend(['#mountain', '#nature', '#landscape'])
        
        if any(word in content_lower for word in ['city', 'urban', 'street', 'building']):
            hashtags.extend(['#city', '#urban', '#street'])
        
        if any(word in content_lower for word in ['food', 'meal', 'restaurant', 'cooking']):
            hashtags.extend(['#food', '#foodie', '#delicious'])
        
        # Add generic hashtags
        hashtags.extend(['#photography', '#instagood', '#picoftheday'])
        
        # Limit number of hashtags
        hashtags = hashtags[:8]
        
        if hashtags:
            return f"{caption}\n\n{' '.join(hashtags)}"
        
        return caption
    
    def _add_to_history(self, caption_result: Dict[str, any]):
        """Add caption to history."""
        self.caption_history.append(caption_result)
        
        # Keep only recent history
        if len(self.caption_history) > self.max_history:
            self.caption_history = self.caption_history[-self.max_history:]
    
    def set_caption_style(self, style: str):
        """Change the caption style."""
        if style in self.style_configs:
            self.caption_style = style
            logging.info(f"Caption style changed to: {style}")
        else:
            logging.warning(f"Unknown caption style: {style}")
    
    def set_update_interval(self, interval: float):
        """Change the update interval."""
        self.update_interval = max(0.5, interval)  # Minimum 0.5 seconds
        logging.info(f"Update interval changed to: {self.update_interval}s")
    
    def get_current_caption(self) -> str:
        """Get the current caption."""
        return self.current_caption
    
    def get_caption_history(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get recent caption history."""
        return self.caption_history[-limit:] if self.caption_history else []
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        if not self.generation_times:
            return {
                "total_generations": 0,
                "avg_generation_time": 0.0,
                "min_generation_time": 0.0,
                "max_generation_time": 0.0
            }
        
        return {
            "total_generations": self.total_generations,
            "avg_generation_time": np.mean(self.generation_times),
            "min_generation_time": np.min(self.generation_times),
            "max_generation_time": np.max(self.generation_times),
            "current_style": self.caption_style,
            "update_interval": self.update_interval
        }
    
    def force_generate_caption(self, frame: np.ndarray) -> Optional[Dict[str, any]]:
        """Force generate a caption immediately."""
        return self.generate_caption(frame, force_update=True)


