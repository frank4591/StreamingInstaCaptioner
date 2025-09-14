import { VideoContext, CaptionResponse, HealthResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export class ApiService {
  static async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }

  static async analyzeFrames(frames: string[]): Promise<VideoContext> {
    const response = await fetch(`${API_BASE_URL}/analyze_frames`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ frames }),
    });

    if (!response.ok) {
      throw new Error('Frame analysis failed');
    }

    return response.json();
  }

  static async generateCaption(
    imageData: string,
    contextFrames: string[] = []
  ): Promise<CaptionResponse> {
    const response = await fetch(`${API_BASE_URL}/generate_video_context_caption`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageData,
        context_frames: contextFrames,
      }),
    });

    if (!response.ok) {
      throw new Error('Caption generation failed');
    }

    return response.json();
  }
}
