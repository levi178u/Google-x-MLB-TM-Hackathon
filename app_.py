from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2 as cv
import numpy as np
from PIL import Image
import requests
from requests.exceptions import RequestException
import tempfile
import os
from datetime import timedelta
import logging
import io
from typing import List, Dict, Any, Tuple
import mimetypes
import magic  # for better file type detection
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 8192
MAX_FRAMES = 5
ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/mpeg', 'video/x-msvideo']
MAX_PROMPT_LENGTH = 500
MIN_FRAME_DIMENSION = 360
MAX_FRAME_DIMENSION = 1920

class VideoAnalysisError(Exception):
    pass

def validate_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def validate_video_type(content_type: str) -> bool:
    return content_type in ALLOWED_VIDEO_TYPES

def analyze_prompt_with_video(frames: List[bytes], prompt: str) -> Dict[str, Any]:

    try:
        analysis_results = {
            "detected_objects": [],
            "scene_description": "",
            "technical_info": {}
        }

        # Sample analysis logic (replace with actual ML models in production)
        if "objects" in prompt.lower():
            analysis_results["detected_objects"] = ["Person", "Ball", "Dog"]
            analysis_results["confidence_scores"] = [0.95, 0.87, 0.82]
        elif "faces" in prompt.lower():
            analysis_results["detected_objects"] = ["Face"]
            analysis_results["face_count"] = 2
            analysis_results["demographics"] = "Analysis disabled for privacy"
        else:
            analysis_results["scene_description"] = "General scene analysis completed"

        return analysis_results
    except Exception as e:
        logger.error(f"Error in prompt analysis: {str(e)}")
        raise VideoAnalysisError(f"Failed to analyze prompt: {str(e)}")

def download_video_in_chunks(url: str) -> Tuple[str, float]:

    if not validate_url(url):
        raise ValueError("Invalid URL format")

    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not validate_video_type(content_type):
                raise ValueError(f"Invalid video type: {content_type}")
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_VIDEO_SIZE:
                raise ValueError(f"Video size exceeds maximum allowed size of {MAX_VIDEO_SIZE/1024/1024}MB")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file_size = 0
            
            try:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file_size += len(chunk)
                        if file_size > MAX_VIDEO_SIZE:
                            temp_file.close()
                            os.unlink(temp_file.name)
                            raise ValueError("Video size exceeded maximum allowed size during download")
                        temp_file.write(chunk)
                
                return temp_file.name, file_size / (1024 * 1024)  # Return path and size in MB
            except Exception:
                temp_file.close()
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise
                
    except RequestException as e:
        raise ValueError(f"Failed to download video: {str(e)}")

def process_video(video_path: str, num_frames: int = MAX_FRAMES) -> Dict[str, Any]:

    cap = None
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoAnalysisError("Failed to open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        duration = total_frames / fps
        
        if total_frames == 0 or fps == 0:
            raise VideoAnalysisError("Invalid video properties detected")
        
        # Calculate frame interval
        interval = max(1, total_frames // num_frames)
        
        frames = []
        timestamps = []
        metadata = {
            "resolution": f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}",
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2)
        }
        
        frame_count = 0
        
        while len(frames) < num_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Validate frame dimensions
                height, width = frame.shape[:2]
                if width < MIN_FRAME_DIMENSION or height < MIN_FRAME_DIMENSION:
                    raise VideoAnalysisError("Frame dimensions too small")
                
                if width > MAX_FRAME_DIMENSION or height > MAX_FRAME_DIMENSION:
                    aspect_ratio = width / height
                    new_width = min(MAX_FRAME_DIMENSION, int(MAX_FRAME_DIMENSION * aspect_ratio))
                    new_height = min(MAX_FRAME_DIMENSION, int(new_width / aspect_ratio))
                    frame = cv.resize(frame, (new_width, new_height))
                
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Compress frame
                frame_byte_arr = io.BytesIO()
                pil_img.save(frame_byte_arr, format='JPEG', quality=85)
                frame_byte_arr.seek(0)
                
                frames.append(frame_byte_arr)
                timestamps.append(str(timedelta(seconds=frame_count / fps)))
            
            frame_count += 1
        
        if not frames:
            raise VideoAnalysisError("No frames could be extracted from the video")
            
        return {
            "frames": frames,
            "timestamps": timestamps,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise VideoAnalysisError(f"Failed to process video: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
 
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        video_url = data.get('videoUrl')
        prompt = data.get('prompt', '')
        analysis_type = data.get('analysisType', 'general')
        
        # Validate inputs
        if not video_url:
            return jsonify({"error": "Video URL is required"}), 400
        if len(prompt) > MAX_PROMPT_LENGTH:
            return jsonify({"error": "Prompt exceeds maximum length"}), 400
            
        # Process video
        temp_video_path, file_size = download_video_in_chunks(video_url)
        
        try:
            result = process_video(temp_video_path)
            analysis = analyze_prompt_with_video(result["frames"], prompt)
            
            response = {
                "status": "success",
                "analysis": {
                    "type": analysis_type,
                    "prompt_result": analysis,
                    "frames": [f"frame_{i+1}" for i in range(len(result["frames"]))],
                    "timestamps": result["timestamps"],
                    "metadata": {
                        **result["metadata"],
                        "file_size_mb": round(file_size, 2)
                    }
                }
            }
            
            return jsonify(response)
            
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                    
    except ValueError as e:
        logger.warning(f"Invalid input: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except VideoAnalysisError as e:
        logger.error(f"Video analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)