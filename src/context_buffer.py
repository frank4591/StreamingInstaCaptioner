"""
Context Buffer for Streaming Instagram Captioner

Manages a sliding window of video context for real-time caption generation.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
import time

class StreamingContextBuffer:
    """
    Manages a sliding window buffer of video context for real-time processing.
    
    Maintains recent frame descriptions and features to provide context
    for Instagram caption generation.
    """
    
    def __init__(
        self,
        max_frames: int = 10,
        context_window_seconds: float = 30.0,
        update_interval: float = 2.0,
        min_context_frames: int = 3
    ):
        """
        Initialize the context buffer.
        
        Args:
            max_frames: Maximum number of frames to keep in buffer
            context_window_seconds: Time window for context (seconds)
            update_interval: How often to update context (seconds)
            min_context_frames: Minimum frames needed for context
        """
        self.max_frames = max_frames
        self.context_window_seconds = context_window_seconds
        self.update_interval = update_interval
        self.min_context_frames = min_context_frames
        
        # Buffer storage
        self.frame_descriptions = deque(maxlen=max_frames)
        self.frame_features = deque(maxlen=max_frames)
        self.frame_timestamps = deque(maxlen=max_frames)
        self.frame_confidence = deque(maxlen=max_frames)
        
        # Context state
        self.current_context = ""
        self.last_update_time = 0.0
        self.context_consistency = 0.0
        
        # Performance tracking
        self.total_updates = 0
        self.context_generation_times = []
        
        logging.info(f"StreamingContextBuffer initialized: {max_frames} frames, {context_window_seconds}s window")
    
    def add_frame(
        self,
        description: str,
        features: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Add a new frame to the context buffer.
        
        Args:
            description: Frame description text
            features: Frame feature vector (optional)
            confidence: Confidence score for the description
            timestamp: Frame timestamp (defaults to current time)
            
        Returns:
            True if context was updated, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add frame data
        self.frame_descriptions.append(description)
        self.frame_features.append(features if features is not None else np.array([]))
        self.frame_timestamps.append(timestamp)
        self.frame_confidence.append(confidence)
        
        # Check if we should update context
        should_update = self._should_update_context(timestamp)
        
        if should_update:
            self._update_context()
            return True
        
        return False
    
    def _should_update_context(self, current_time: float) -> bool:
        """Check if context should be updated."""
        # Always update if we have enough frames and enough time has passed
        time_since_update = current_time - self.last_update_time
        
        return (
            len(self.frame_descriptions) >= self.min_context_frames and
            time_since_update >= self.update_interval
        )
    
    def _update_context(self):
        """Update the current context from the buffer."""
        start_time = time.time()
        
        try:
            if len(self.frame_descriptions) < self.min_context_frames:
                self.current_context = "Insufficient context available"
                return
            
            # Filter frames by time window
            current_time = time.time()
            recent_frames = self._get_recent_frames(current_time)
            
            if not recent_frames:
                self.current_context = "No recent context available"
                return
            
            # Aggregate context from recent frames
            self.current_context = self._aggregate_context(recent_frames)
            
            # Calculate context consistency
            self.context_consistency = self._calculate_consistency(recent_frames)
            
            # Update tracking
            self.last_update_time = current_time
            self.total_updates += 1
            
            processing_time = time.time() - start_time
            self.context_generation_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self.context_generation_times) > 100:
                self.context_generation_times = self.context_generation_times[-100:]
            
            logging.debug(f"Context updated: {len(recent_frames)} frames, {processing_time:.3f}s")
            
        except Exception as e:
            logging.error(f"Error updating context: {str(e)}")
            self.current_context = "Error updating context"
    
    def _get_recent_frames(self, current_time: float) -> List[Dict]:
        """Get frames within the context window."""
        recent_frames = []
        
        for i in range(len(self.frame_timestamps)):
            frame_time = self.frame_timestamps[i]
            time_diff = current_time - frame_time
            
            if time_diff <= self.context_window_seconds:
                recent_frames.append({
                    'description': self.frame_descriptions[i],
                    'features': self.frame_features[i],
                    'confidence': self.frame_confidence[i],
                    'timestamp': frame_time,
                    'age': time_diff
                })
        
        return recent_frames
    
    def _aggregate_context(self, frames: List[Dict]) -> str:
        """Aggregate context from multiple frames."""
        if not frames:
            return "No context available"
        
        # Simple weighted aggregation based on confidence and recency
        descriptions = []
        weights = []
        
        for frame in frames:
            descriptions.append(frame['description'])
            
            # Weight based on confidence and recency
            confidence_weight = frame['confidence']
            recency_weight = max(0, 1.0 - (frame['age'] / self.context_window_seconds))
            weight = confidence_weight * recency_weight
            
            weights.append(weight)
        
        # Select top descriptions by weight
        if len(descriptions) <= 3:
            selected_descriptions = descriptions
        else:
            # Get top 3 descriptions by weight
            frame_weights = list(zip(descriptions, weights))
            frame_weights.sort(key=lambda x: x[1], reverse=True)
            selected_descriptions = [desc for desc, _ in frame_weights[:3]]
        
        # Combine selected descriptions
        context = " ".join(selected_descriptions)
        
        # Truncate if too long
        max_length = 500
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context
    
    def _calculate_consistency(self, frames: List[Dict]) -> float:
        """Calculate consistency score for the context."""
        if len(frames) < 2:
            return 1.0
        
        try:
            # Simple consistency based on common words
            descriptions = [frame['description'] for frame in frames]
            
            # Calculate word overlap
            word_sets = [set(desc.lower().split()) for desc in descriptions]
            
            if not word_sets:
                return 0.0
            
            # Calculate pairwise Jaccard similarity
            similarities = []
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    intersection = len(word_sets[i].intersection(word_sets[j]))
                    union = len(word_sets[i].union(word_sets[j]))
                    
                    if union > 0:
                        similarity = intersection / union
                        similarities.append(similarity)
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Error calculating consistency: {str(e)}")
            return 0.5
    
    def get_current_context(self) -> Dict[str, any]:
        """Get the current context information."""
        return {
            "context_text": self.current_context,
            "consistency": self.context_consistency,
            "frame_count": len(self.frame_descriptions),
            "last_update": self.last_update_time,
            "total_updates": self.total_updates
        }
    
    def get_buffer_stats(self) -> Dict[str, any]:
        """Get buffer statistics."""
        current_time = time.time()
        
        return {
            "total_frames": len(self.frame_descriptions),
            "buffer_utilization": len(self.frame_descriptions) / self.max_frames,
            "oldest_frame_age": current_time - self.frame_timestamps[0] if self.frame_timestamps else 0,
            "newest_frame_age": current_time - self.frame_timestamps[-1] if self.frame_timestamps else 0,
            "avg_confidence": np.mean(self.frame_confidence) if self.frame_confidence else 0,
            "context_consistency": self.context_consistency,
            "avg_context_generation_time": np.mean(self.context_generation_times) if self.context_generation_times else 0
        }
    
    def clear_buffer(self):
        """Clear the context buffer."""
        self.frame_descriptions.clear()
        self.frame_features.clear()
        self.frame_timestamps.clear()
        self.frame_confidence.clear()
        
        self.current_context = ""
        self.context_consistency = 0.0
        self.last_update_time = 0.0
        
        logging.info("Context buffer cleared")
    
    def force_update_context(self):
        """Force an immediate context update."""
        self._update_context()
    
    def is_ready(self) -> bool:
        """Check if buffer has enough context for caption generation."""
        return (
            len(self.frame_descriptions) >= self.min_context_frames and
            self.current_context != ""
        )


