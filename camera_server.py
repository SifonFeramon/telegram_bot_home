# -*- coding: utf-8 -*-
"""
Camera Server - REST API for capturing images from RTSP camera stream.

This server connects to an IP camera via RTSP and exposes HTTP endpoints
to capture images on demand.

Usage:
    python camera_server.py -rtsp_url "rtsp://camera_ip:554/stream" -port 8080
"""

import argparse
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Camera Server",
    description="REST API for capturing images from RTSP camera stream",
    version="1.0.0"
)

# Global configuration
rtsp_url: str = ""
camera_lock = threading.Lock()


class CameraManager:
    """Manages camera connection with reconnection logic."""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0
        self._frame_cache_ttl: float = 0.5  # Cache frame for 0.5 seconds
    
    def _connect(self) -> bool:
        """Establish connection to camera."""
        try:
            if self._cap is not None:
                self._cap.release()
            
            self._cap = cv2.VideoCapture(self.rtsp_url)
            if self._cap.isOpened():
                logger.info(f"Connected to camera: {self.rtsp_url}")
                return True
            else:
                logger.error(f"Failed to open camera: {self.rtsp_url}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        with self._lock:
            # Return cached frame if still valid
            current_time = time.time()
            if (self._last_frame is not None and 
                current_time - self._last_frame_time < self._frame_cache_ttl):
                return self._last_frame.copy()
            
            # Try to read frame, reconnect if necessary
            for attempt in range(3):
                if self._cap is None or not self._cap.isOpened():
                    if not self._connect():
                        continue
                
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._last_frame = frame
                    self._last_frame_time = current_time
                    return frame.copy()
                else:
                    logger.warning(f"Failed to read frame, attempt {attempt + 1}/3")
                    self._connect()
            
            return None
    
    def release(self):
        """Release camera resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


# Global camera manager instance
camera_manager: Optional[CameraManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize camera manager on startup."""
    global camera_manager
    camera_manager = CameraManager(rtsp_url)
    logger.info("Camera server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Release camera resources on shutdown."""
    global camera_manager
    if camera_manager:
        camera_manager.release()
    logger.info("Camera server stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rtsp_url": rtsp_url[:50] + "..." if len(rtsp_url) > 50 else rtsp_url
    }


@app.get("/capture")
async def capture_image(quality: int = 85):
    """
    Capture current frame from camera and return as JPEG image.
    
    Args:
        quality: JPEG quality (1-100), default 85
    
    Returns:
        JPEG image bytes
    """
    if camera_manager is None:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    
    frame = camera_manager.capture_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame from camera")
    
    # Encode frame as JPEG
    quality = max(1, min(100, quality))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode('.jpg', frame, encode_params)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=capture.jpg"}
    )


@app.get("/capture/png")
async def capture_image_png():
    """
    Capture current frame from camera and return as PNG image.
    
    Returns:
        PNG image bytes
    """
    if camera_manager is None:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    
    frame = camera_manager.capture_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame from camera")
    
    # Encode frame as PNG
    success, encoded = cv2.imencode('.png', frame)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=capture.png"}
    )


def main():
    global rtsp_url
    
    parser = argparse.ArgumentParser(description="Camera Server - REST API for RTSP camera")
    parser.add_argument(
        "-rtsp_url", 
        dest="rtsp_url", 
        type=str, 
        required=True,
        help="RTSP URL of the camera (e.g., rtsp://192.168.1.100:554/stream)"
    )
    parser.add_argument(
        "-port", 
        dest="port", 
        type=int, 
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "-host", 
        dest="host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    rtsp_url = args.rtsp_url
    
    logger.info(f"Starting camera server on {args.host}:{args.port}")
    logger.info(f"RTSP URL: {rtsp_url}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
