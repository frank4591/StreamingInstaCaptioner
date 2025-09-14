import { useState, useRef, useCallback } from 'react';
import { CameraState } from '../types';

export const useCamera = () => {
  const [cameraState, setCameraState] = useState<CameraState>({
    isStreaming: false,
    isContextCollecting: false,
    currentInterval: 2,
    maxContextFrames: 10,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('Got stream:', stream);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        return new Promise<void>((resolve, reject) => {
          videoRef.current!.onloadedmetadata = () => {
            console.log('Video metadata loaded');
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
              console.log('Canvas size set:', canvasRef.current.width, canvasRef.current.height);
            }
            setCameraState(prev => ({ ...prev, isStreaming: true }));
            console.log('Camera state updated to streaming');
            resolve();
          };
          
          videoRef.current!.onerror = (error) => {
            console.error('Video error:', error);
            reject(error);
          };
        });
      } else {
        console.error('Video ref is null');
        throw new Error('Video ref is null');
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      throw error;
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    setCameraState(prev => ({
      ...prev,
      isStreaming: false,
      isContextCollecting: false,
    }));
  }, []);

  const captureFrame = useCallback((): string | null => {
    console.log('captureFrame called');
    if (!videoRef.current || !canvasRef.current) {
      console.log('Video or canvas ref is null');
      return null;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      console.log('Canvas context is null');
      return null;
    }

    console.log('Drawing image to canvas, video dimensions:', video.videoWidth, video.videoHeight);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    console.log('Frame captured, data URL length:', dataUrl.length);
    return dataUrl;
  }, []);

  const startContextCollection = useCallback((onFrameCapture: (frameData: string) => void, intervalSeconds?: number) => {
    console.log('startContextCollection called, camera streaming:', cameraState.isStreaming);
    
    if (!cameraState.isStreaming) {
      console.log('Cannot start context collection: camera not streaming');
      return;
    }

    // Clear any existing interval first
    if (intervalRef.current) {
      console.log('Clearing existing interval');
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Use provided interval or current state interval
    const intervalToUse = intervalSeconds || cameraState.currentInterval;
    console.log('Starting context collection with interval:', intervalToUse);
    setCameraState(prev => ({ ...prev, isContextCollecting: true }));

    // Start immediately
    console.log('Capturing first frame immediately...');
    const firstFrame = captureFrame();
    if (firstFrame) {
      console.log('First frame captured, calling onFrameCapture');
      onFrameCapture(firstFrame);
    } else {
      console.log('First frame capture failed');
    }

    // Then start interval - use the provided or current interval value
    const intervalMs = intervalToUse * 1000;
    console.log('Setting up interval for', intervalMs, 'ms');
    
    intervalRef.current = setInterval(() => {
      console.log('Context collection interval triggered at', new Date().toLocaleTimeString());
      const frameData = captureFrame();
      if (frameData) {
        console.log('Frame captured, calling onFrameCapture');
        onFrameCapture(frameData);
      } else {
        console.log('No frame data captured');
      }
    }, intervalMs);
    
    console.log('Interval set up with ID:', intervalRef.current);
  }, [cameraState.isStreaming, captureFrame]);

  const stopContextCollection = useCallback(() => {
    console.log('Stopping context collection, interval ID:', intervalRef.current);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      console.log('Interval cleared');
    }

    setCameraState(prev => ({ ...prev, isContextCollecting: false }));
  }, []);

  // Debug function to check interval status
  const checkIntervalStatus = useCallback(() => {
    console.log('Interval status check:', {
      intervalId: intervalRef.current,
      isContextCollecting: cameraState.isContextCollecting,
      isStreaming: cameraState.isStreaming,
      currentInterval: cameraState.currentInterval
    });
  }, [cameraState]);

  const setIntervalValue = useCallback((interval: number) => {
    setCameraState(prev => ({ ...prev, currentInterval: interval }));
  }, []);

  return {
    cameraState,
    videoRef,
    canvasRef,
    startCamera,
    stopCamera,
    captureFrame,
    startContextCollection,
    stopContextCollection,
    setInterval: setIntervalValue,
    checkIntervalStatus,
  };
};
