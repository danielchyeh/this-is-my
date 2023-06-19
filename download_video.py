import os
import json
import argparse
import pandas as pd
import numpy as np
import csv
from pytube import YouTube
import pytube
from moviepy.editor import VideoFileClip
from thisismy_utils import parse_dataset,load_thisismy

# Function to read the CSV file and return a dictionary of segment IDs
def read_csv(csv_file):
    segment_id_map = {}
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            segment_id = row['segment_id']
            video_id = row['video_id']
            start_time = float(row['start_time'])
            end_time = float(row['end_time'])
            
            segment_id_map[segment_id] = {'video_id': video_id, 'start_time': start_time, 'end_time': end_time}
    
    return segment_id_map

def check_video_existence(video_id, folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith(video_id):
            return True
    return False

def download_video(segment_id, video_id, start_time, end_time, output_video, output_segment):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    youtube = YouTube(youtube_url)
    try:
        if check_video_existence(video_id, output_video):
            video_path = f"{output_video}/{video_id}_{youtube.title}.mp4"
            print("Video already exists")
        else:
            video = youtube.streams.filter(progressive=True, file_extension="mp4").first()
            video_path = video.download(output_path=output_video, filename=f"{video_id}_{youtube.title}.mp4")

    except pytube.exceptions.VideoPrivate:
        print(f"Skipping private video: {video_id}")
        return

    output_filename = f"{output_segment}/{segment_id}.mp4"
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    clip.write_videofile(output_filename, codec="libx264")
    print(f"Video downloaded and saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default='this-is-my-dataset/', help='access data/label')
    parser.add_argument('--ANNO_DIR', type=str, default='this-is-my_test-set.json', help='access label file')
    parser.add_argument('--SEG_DIR', type=str, default='segments.csv', help='access csv file')
    parser.add_argument('--MODE', type=str, default='eval', choices=['train', 'eval'], help='download on train or eval set')
    args = parser.parse_args()

    ANNO_FILE = os.path.join(args.DATA_DIR, args.ANNO_DIR)
    SEGMENT_FILE = os.path.join(args.DATA_DIR, args.SEG_DIR)
    OUTPUT_SEG, OUTPUT_VIDEO = f'{args.MODE}_segment', f'{args.MODE}_video'

    train_x, train_y, eval_x, eval_y, train_class, eval_class, token2class, id2classname, token2item = load_thisismy(ANNO_FILE,SEGMENT_FILE)

    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_SEG, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO, exist_ok=True)

    # Read the CSV file and create a dictionary mapping segment IDs to video IDs, start times, and end times
    segment_id_map = read_csv(SEGMENT_FILE)

    if args.MODE == 'train': 
        download_x = train_x
    else: 
        download_x = eval_x

    # Iterate over the segment IDs in train_x and retrieve the corresponding video ID, start time, and end time
    for segment_id in download_x:
        if segment_id in segment_id_map:
            video_id = segment_id_map[segment_id]['video_id']
            start_time = segment_id_map[segment_id]['start_time']
            end_time = segment_id_map[segment_id]['end_time']
            
            # Download the video from YouTube based on the start and end time
            download_video(segment_id, video_id, start_time, end_time, OUTPUT_VIDEO, OUTPUT_SEG)
            
            print(f"Segment ID: {segment_id}, Video ID: {video_id}, Start Time: {start_time}, End Time: {end_time}")
        else:
            print(f"No matching entry found for Segment ID: {segment_id}")
