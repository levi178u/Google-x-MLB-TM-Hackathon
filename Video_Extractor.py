import uuid
import requests
import cv2 as cv
import pandas as pd
from PIL import Image
from datetime import timedelta
from tqdm import tqdm
import csv

def replace_video_with_images(text, frames):
    return text.replace("<video>", "<image>" * frames)

def sample_frames(url, num_frames):
    try:
        response = requests.get(url)
        response.raise_for_status()

        path_id = str(uuid.uuid4())
        path = f"./{path_id}.mp4"
        
        with open(path, "wb") as f:
            f.write(response.content)

        video = cv.VideoCapture(path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video from URL: {url}")
        
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv.CAP_PROP_FPS)
        duration = total_frames / fps

        interval = total_frames // num_frames
        frames = []
        timestamps = []

        for i in range(total_frames):
            ret, frame = video.read()
            if not ret:
                continue
            if i % interval == 0:
                pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                frames.append(pil_img)
                timestamp = (i / total_frames) * duration
                timestamps.append(str(timedelta(seconds=timestamp)))

        video.release()
        return frames[:num_frames], timestamps, duration

    except requests.exceptions.RequestException as e:
        print(f"Error downloading video from URL {url}: {e}")
        return [], [], 0
    except cv.error as e:
        print(f"OpenCV error with video URL {url}: {e}")
        return [], [], 0
    except ValueError as e:
        print(f"Error processing video from URL {url}: {e}")
        return [], [], 0
    except Exception as e:
        print(f"Unexpected error occurred with video URL {url}: {e}")
        return [], [], 0

def process_csv(input_csv, url_column, max_videos=None, num_frames=5):
    try:
        df = pd.read_csv(input_csv)
        
        frame_data = []
        duration_data = []
        timestamp_data = []

        video_count = 0
        total_rows = len(df)
        
        for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Videos", unit="video"):
            if max_videos and video_count >= max_videos:
                break
            
            video_url = row[url_column]
            
            frames, timestamps, duration = sample_frames(video_url, num_frames)
            if frames:
                frame_data.append(frames)
                timestamp_data.append(timestamps)
                duration_data.append(duration)
            else:
                frame_data.append([])
                timestamp_data.append([])
                duration_data.append(0)

            video_count += 1

            print(f"Processed Video {video_count}/{total_rows} - Duration: {duration:.2f}s - Frames: {len(frames)} - Columns: {df.shape[1]}")

        df['Frames (image objects)'] = frame_data
        df['Video Duration (seconds)'] = duration_data
        df['Timestamps (for frames)'] = timestamp_data

        df.to_csv(input_csv, index=False)
        print(f"Processing complete. The updated data is saved in {input_csv}.")

    except pd.errors.EmptyDataError:
        print(f"Error: The input CSV file '{input_csv}' is empty.")
    except FileNotFoundError:
        print(f"Error: The input CSV file '{input_csv}' was not found.")

input_csv = '/home/ichigo/Documents/GitHub/GoogleMLBCompetition/Model_Section/2024-mlb-homeruns.csv'
url_column = 'video'
max_videos = 1

try:
    num_frames = int(input("Enter the number of frames to extract per video: "))
    if num_frames <= 0:
        raise ValueError("Number of frames must be a positive integer.")
except ValueError as e:
    print(f"Invalid input for number of frames: {e}")
    num_frames = 26

process_csv(input_csv, url_column, max_videos, num_frames)
