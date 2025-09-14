#!/usr/bin/env python3
"""
Video Context Server - Uses actual VideoContextImageCaptioning pipeline
This server directly uses the VideoContextImageCaptioning pipeline for proper context extraction
"""

import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import base64
import io
import time
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image

# Add the VideoContextImageCaptioning project root to path
video_context_path = Path("/home/frank/VideoContextImageCaptioning")
if video_context_path.exists():
    sys.path.insert(0, str(video_context_path))
    try:
        # Import from the src package
        from src.pipeline import VideoContextCaptionPipeline
        from src.video_processor import VideoProcessor
        from src.context_extractor import ContextExtractor
        from src.liquid_integration import LiquidAIIntegration
        VIDEO_CONTEXT_AVAILABLE = True
        logging.info("Video context modules loaded successfully")
    except ImportError as e:
        logging.warning(f"Video context modules not available: {e}")
        VIDEO_CONTEXT_AVAILABLE = False
else:
    logging.warning(f"VideoContextImageCaptioning project not found at {video_context_path}")
    VIDEO_CONTEXT_AVAILABLE = False

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
pipeline = None
video_processor = None
context_extractor = None
liquid_integration = None
device = None

class CaptionRequest(BaseModel):
    image_data: str
    context: Optional[str] = None
    style: str = "instagram"
    prompt: Optional[str] = None

class VideoContextRequest(BaseModel):
    frames: List[str]  # List of base64 encoded images
    current_frame: str  # Current frame for captioning
    style: str = "instagram"
    prompt: Optional[str] = None

class FrameAnalysisRequest(BaseModel):
    frames: List[str]  # List of base64 encoded images

class CaptionResponse(BaseModel):
    caption: str
    instagram_caption: str
    raw_output: str
    context: str
    prompt: str
    confidence: float
    processing_time: float

class VideoContextResponse(BaseModel):
    caption: str
    instagram_caption: str
    raw_output: str
    video_context: str
    frame_descriptions: List[str]
    context_consistency: float
    confidence: float
    processing_time: float

class FrameAnalysisResponse(BaseModel):
    frame_descriptions: List[str]
    aggregated_context: str
    frame_features: List[Dict]
    processing_time: float

def check_gpu_availability():
    """Check GPU availability and set device."""
    global device
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"   GPU {i}: {gpu_name}")
        
        # Try to use GPU 1 if available, otherwise GPU 0
        if gpu_count > 1:
            device = "cuda:1"
            torch.cuda.set_device(1)
            logger.info(f"Using GPU 1 for model inference")
        else:
            device = "cuda:0"
            torch.cuda.set_device(0)
            logger.info(f"Using GPU 0 for model inference")
    else:
        device = "cpu"
        logger.info("⚠️  CUDA not available, using CPU")
    
    return device

def load_video_context_components():
    """Load the VideoContextImageCaptioning components."""
    global pipeline, video_processor, context_extractor, liquid_integration, device
    
    if not VIDEO_CONTEXT_AVAILABLE:
        logger.warning("Video context modules not available, using fallback mode")
        return False
    
    try:
        model_path = "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"
        
        logger.info("Initializing VideoContextImageCaptioning components...")
        
        # Initialize individual components like in the pipeline
        video_processor = VideoProcessor(
            strategy="key_frames",
            max_frames=10
        )
        
        context_extractor = ContextExtractor(
            aggregation_method="weighted_average"
        )
        
        liquid_integration = LiquidAIIntegration(
            model_path=model_path,
            device=device
        )
        
        # Initialize full pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path=model_path,
            device=device,
            frame_extraction_strategy="key_frames",
            max_frames=10,
            context_aggregation="weighted_average",
            context_weight=0.7
        )
        
        logger.info(f"✅ VideoContextImageCaptioning components loaded successfully")
        logger.info(f"Components device: {device}")
        
        # Test the components
        logger.info("Testing video context components...")
        test_image = Image.new('RGB', (224, 224), color='white')
        # Convert PIL Image to numpy array for testing
        test_array = np.array(test_image)
        # Convert RGB to BGR to match OpenCV format
        test_array = cv2.cvtColor(test_array, cv2.COLOR_RGB2BGR)
        test_result = liquid_integration.generate_caption(
            image=test_array,
            return_features=True
        )
        logger.info("✅ Video context components test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load video context components: {e}")
        logger.warning("Continuing with fallback mode")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline, device
    
    logger.info("🚀 Starting Video Context Server...")
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Load video context components
    success = load_video_context_components()
    
    if success:
        logger.info("✅ Video Context Server ready with full functionality!")
    else:
        logger.info("⚠️  Video Context Server ready with limited functionality")
    
    yield
    
    logger.info("🛑 Shutting down Video Context Server...")

# Create FastAPI app
app = FastAPI(
    title="Video Context Server API",
    description="API for video context extraction using VideoContextImageCaptioning pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_data: str) -> Image.Image:
    """Preprocess base64 image data."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def create_temp_video(frames: List[Image.Image]) -> str:
    """Create a temporary video file from frames."""
    if not frames:
        raise ValueError("No frames provided")
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    # Get video dimensions from first frame
    height, width = frames[0].size[1], frames[0].size[0]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, 1.0, (width, height))
    
    try:
        for frame in frames:
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        
        out.release()
        return temp_video.name
        
    except Exception as e:
        out.release()
        os.unlink(temp_video.name)
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "video_context_available": VIDEO_CONTEXT_AVAILABLE,
        "device": device,
        "timestamp": time.time()
    }

@app.post("/analyze_frames", response_model=FrameAnalysisResponse)
async def analyze_frames(request: FrameAnalysisRequest):
    """Analyze frames using VideoContextImageCaptioning pipeline."""
    try:
        if not VIDEO_CONTEXT_AVAILABLE or not liquid_integration or not context_extractor:
            raise HTTPException(
                status_code=503, 
                detail="Video context components not available"
            )
        
        start_time = time.time()
        
        # Preprocess frames
        frames = []
        for frame_data in request.frames:
            try:
                frame_image = preprocess_image(frame_data)
                frames.append(frame_image)
            except Exception as e:
                logger.warning(f"Failed to process frame: {e}")
                continue
        
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames provided")
        
        # Process frames using VideoContextImageCaptioning pipeline
        frame_captions = []
        frame_features = []
        frame_features_for_response = []
        
        for i, frame in enumerate(frames):
            logger.info(f"Processing frame {i+1}/{len(frames)}")
            
            # Convert PIL Image to numpy array in the format expected by video processor
            # Video processor returns frames as BGR numpy arrays from OpenCV
            frame_array = np.array(frame)
            
            # Convert RGB to BGR to match OpenCV format (video processor uses BGR)
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            
            # Use the same method as in VideoContextImageCaptioning pipeline
            caption_result = liquid_integration.generate_caption(
                image=frame_array,
                return_features=True
            )
            
            # Clean the caption by removing prompt text
            caption = caption_result["caption"]
            # Remove common prompt prefixes - handle various patterns
            if "user Describe this image in detail. assistant" in caption:
                # Remove the full prompt and clean up
                caption = caption.replace("user Describe this image in detail. assistant", "").strip()
                # Remove any leading "In the image," if it remains
                if caption.startswith("In the image,"):
                    caption = caption.replace("In the image,", "").strip()
            elif "assistant In the image," in caption:
                caption = caption.replace("assistant In the image,", "").strip()
            elif "In the image," in caption:
                caption = caption.replace("In the image,", "").strip()
            elif caption.startswith("assistant "):
                caption = caption.replace("assistant ", "").strip()
            
            frame_captions.append(caption)
            
            # Store features for context extractor (numpy arrays)
            if isinstance(caption_result["features"], np.ndarray):
                frame_features.append(caption_result["features"])
                frame_features_for_response.append({"features": caption_result["features"].tolist()})
            else:
                # If features are not numpy array, create a dummy array
                dummy_features = np.random.randn(768)  # Standard feature size
                frame_features.append(dummy_features)
                frame_features_for_response.append({"features": dummy_features.tolist()})
        
        # Aggregate context using the same method as in VideoContextImageCaptioning pipeline
        aggregated_context = context_extractor.aggregate_context(
            frame_captions=frame_captions,
            frame_features=frame_features,
            max_length=512
        )
        
        processing_time = time.time() - start_time
        
        return FrameAnalysisResponse(
            frame_descriptions=frame_captions,
            aggregated_context=aggregated_context["context_text"],
            frame_features=frame_features_for_response,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    """Generate a caption for a single image."""
    try:
        if not liquid_integration:
            raise HTTPException(
                status_code=503, 
                detail="Liquid integration not available"
            )
        
        # Preprocess image
        image = preprocess_image(request.image_data)
        
        # Generate caption using liquid integration
        start_time = time.time()
        
        result = liquid_integration.generate_caption(
            image=image,
            text_prompt=request.prompt
        )
        
        processing_time = time.time() - start_time
        
        return CaptionResponse(
            caption=result.get("caption", "No caption generated"),
            instagram_caption=result.get("instagram_caption", result.get("caption", "No caption generated")),
            raw_output=result.get("raw_output", ""),
            context=request.context or "",
            prompt=request.prompt or "",
            confidence=result.get("confidence", 0.8),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_video_context_caption", response_model=VideoContextResponse)
async def generate_video_context_caption(request: VideoContextRequest):
    """Generate a caption using video context from multiple frames."""
    try:
        if not pipeline:
            raise HTTPException(
                status_code=503, 
                detail="Video context pipeline not available"
            )
        
        # Preprocess current frame
        current_image = preprocess_image(request.current_frame)
        
        # Preprocess context frames
        context_frames = []
        for frame_data in request.frames:
            try:
                frame_image = preprocess_image(frame_data)
                context_frames.append(frame_image)
            except Exception as e:
                logger.warning(f"Failed to process frame: {e}")
                continue
        
        if not context_frames:
            raise HTTPException(status_code=400, detail="No valid context frames provided")
        
        # Create temporary video from frames
        temp_video_path = None
        try:
            temp_video_path = create_temp_video(context_frames)
            
            # Save current image temporarily
            temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            current_image.save(temp_image_path.name)
            temp_image_path.close()
            
            # Use the full pipeline
            start_time = time.time()
            
            result = pipeline.generate_caption(
                image_path=temp_image_path.name,
                video_path=temp_video_path,
                text_prompt=request.prompt
            )
            
            processing_time = time.time() - start_time
            
            # Clean up temporary files
            os.unlink(temp_image_path.name)
            
            return VideoContextResponse(
                caption=result.get("caption", "No caption generated"),
                instagram_caption=result.get("instagram_caption", result.get("caption", "No caption generated")),
                raw_output=result.get("raw_output", ""),
                video_context=result.get("video_context", {}).get("context_text", ""),
                frame_descriptions=result.get("video_context", {}).get("frame_captions", []),
                context_consistency=result.get("video_context", {}).get("temporal_consistency", 0.0),
                confidence=result.get("confidence", 0.8),
                processing_time=processing_time
            )
            
        finally:
            # Clean up temporary video
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
    except Exception as e:
        logger.error(f"Video context caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
