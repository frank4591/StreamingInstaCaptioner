export interface FrameInfo {
  imageData: string;
  timestamp: number;
  description: string | null;
  quality: number;
}

export interface VideoContext {
  frameDescriptions: string[];
  aggregatedContext: string;
  frameFeatures: Array<{ features: number[] }>;
  processingTime: number;
}

export interface CaptionResponse {
  instagram_caption: string;
  caption: string;
  context: string;
  confidence: number;
  processing_time: number;
}

export interface HealthResponse {
  status: string;
  pipeline_ready: boolean;
  video_context_available: boolean;
  device: string;
  timestamp: number;
}

export interface Stats {
  totalCaptions: number;
  avgProcessingTime: number;
  smartScore: number;
  contextQuality: number;
}

export interface CameraState {
  isStreaming: boolean;
  isContextCollecting: boolean;
  currentInterval: number;
  maxContextFrames: number;
}
