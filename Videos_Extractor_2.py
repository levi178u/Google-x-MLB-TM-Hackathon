import uuid
import requests
import cv2 as cv
from PIL import Image
import csv
from tqdm import tqdm

def replace_video_with_images(text, frames):
    return text.replace("<video>", "<image>" * frames)

def sample_frames(url, num_frames):
    response = requests.get(url)
    path_id = str(uuid.uuid4())

    path = f"./{path_id}.mp4"

    with open(path, "wb") as f:
        f.write(response.content)

    video = cv.VideoCapture(path)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    interval = total_frames // num_frames
    frames = []
    timestamps = []
    
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        if not ret:
            continue
        
        if i % interval == 0:
            
            timestamp = i / fps
            frames.append(pil_img)
            timestamps.append(timestamp)
            
    video.release()
    return list(zip(frames[:num_frames], timestamps[:num_frames]))

def extract_video_urls_from_csv(csv_file_path, column_name, max_videos=None):
    video_urls = []
    
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            video_urls.append(row[column_name])
            if max_videos and len(video_urls) >= max_videos:
                break 
            
    return video_urls

def process_video_urls(csv_file_path, column_name, num_frames, max_videos=None):
    video_urls = extract_video_urls_from_csv(csv_file_path, column_name, max_videos)

    processed_videos = []
    
    for url in tqdm(video_urls, desc="Processing Videos", unit="video"):
        
        frames = sample_frames(url, num_frames)
        # print(f"Processing video: {url}, Frames extracted: {len(frames)}")   
        text = "<video>"  
        image_text = replace_video_with_images(text, len(frames))  
        processed_videos.append((image_text, frames))  
    
    print(f"Total processed videos: {len(processed_videos)}")
    return processed_videos


def add_output_to_csv(csv_file_path, column_name, num_frames, output_column_name, max_videos=None):
    video_urls = extract_video_urls_from_csv(csv_file_path, column_name, max_videos)
    
    processed_videos = process_video_urls(csv_file_path, column_name, num_frames, max_videos)
    
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [output_column_name]  # Add new column to fieldnames
        
        rows = list(reader)
        print(f"Total rows in CSV: {len(rows)}")
        
    if len(rows) != len(processed_videos):
        print(f"Warning: CSV row count ({len(rows)}) does not match processed video count ({len(processed_videos)}).")
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = reader.fieldnames + [output_column_name]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(rows):
            
            if idx < len(processed_videos): 
                image_text, frames_with_timestamps = processed_videos[idx]
                row[output_column_name] = image_text
                writer.writerow(row)
            else:
                print(f"Warning: Skipping row {idx}, no processed video available.") 




#############################################################

# Example usage:
csv_file_path = '/home/ichigo/Documents/GitHub/GoogleMLBCompetition/Model_Section/Datasets/2024-mlb-homeruns.csv'  
column_name = 'video' 
output_column_name = 'processed_video_text'  # Name of the new column in the CSV
num_frames = 26  # Number of frames
max_videos = 1  # Specify how many videos



add_output_to_csv(csv_file_path, column_name, num_frames, output_column_name, max_videos)

print(f"Processed data has been added to the column '{output_column_name}' in the CSV file.")

processed_videos = process_video_urls(csv_file_path, column_name, num_frames, max_videos)

if processed_videos:
    image_text, frames_with_timestamps = processed_videos[0]
    print("Image Text:", image_text)
    for idx, (frame, timestamp) in enumerate(frames_with_timestamps):
        print(f"Frame {idx + 1}: Timestamp = {timestamp} seconds")
else:
    print("No videos were processed.")