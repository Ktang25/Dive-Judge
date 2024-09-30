import os
import cv2
import base64
import json
import pandas as pd
import re
import time
from openai import OpenAI, OpenAIError

# API key for OpenAI access
api_key = "ReplaceWithYourAPIKey"
NumberOfSamples = 0  # Number of example samples to load

if not api_key:
    raise ValueError("API key is not set")

client = OpenAI(api_key=api_key)

# Path to video and its actual score
video_file_path = "VideoToBeScore"
actual_score = "ActualScoreOfVideo"

# Initial prompt to guide GPT model
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video of a diving performance. Below are examples with scores. "
            "The score is calculated based on takeoff, flight, and entry. A poor aspect lowers the overall score."
        ],
    }
]

start_time = time.time()

# Convert video frames to base64-encoded format
def process_video(file_path):
    base64_frames = []
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {file_path}")
        return None

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return base64_frames

# Select every nth frame to limit data size
def limit_frames(frames, step):
    return frames[0::step]

# Load example dives from CSV and append them to the prompt
def load_example_dives():
    global NumberOfSamples

    example_csv_file_path = "/Users/ktang25/PycharmProjects/DiveJudge/103B Dives.csv"
    example_video_folder_path = "/Users/ktang25/PycharmProjects/DiveJudge/103B"
    example_df = pd.read_csv(example_csv_file_path)

    frames = []
    for idx, row in example_df.iterrows():
        if NumberOfSamples > 0:
            filename = row["File"]
            file_path = os.path.join(example_video_folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith((".mov", ".mp4")):
                base64_frames = process_video(file_path)
                if base64_frames:
                    frames.append({
                        "base64Frames": base64_frames,
                        "Takeoff": row["Takeoff"],
                        "Flight": row["Flight"],
                        "Entry": row["Entry"],
                        "Score": row["Score"]
                    })
                else:
                    print(f"Error: Failed to process video file {file_path}")
            NumberOfSamples -= 1

    MAX_SIZE_MB = 20
    total_size = 0

    for idx, example in enumerate(frames):
        limited_frames = limit_frames(example["base64Frames"], 30)
        frame_size = sum(len(x) for x in limited_frames) / (1024 * 1024)  # Convert to MB
        if total_size + frame_size > MAX_SIZE_MB:
            break
        total_size += frame_size

        PROMPT_MESSAGES[0]["content"].extend([
            f"Example {idx + 1}:",
            *map(lambda x: {"image": x, "resize": 258}, limited_frames),
            f"Takeoff: {example['Takeoff']}",
            f"Flight: {example['Flight']}",
            f"Entry: {example['Entry']}",
            f"Score: {example['Score']}",
            ""
        ])

# Extract numerical score from GPT response
def extract_score_from_response(response_text):
    scores = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
    if scores:
        return float(scores[-1])
    else:
        raise ValueError("No numerical score found in the response.")

# Send dive video frames to GPT and get the predicted score
def score_dive(file_name, base64_frames):
    global PROMPT_MESSAGES

    if base64_frames is None or len(base64_frames) == 0:
        print(f"Error: No frames to process for {file_name}")
        return None

    limited_frames = limit_frames(base64_frames, 30)
    PROMPT_MESSAGES[0]["content"].append(
        f"Please evaluate the following frames from the diving performance in {file_name} and provide a score:"
    )
    PROMPT_MESSAGES[0]["content"].extend(map(lambda x: {"image": x, "resize": 258}, limited_frames))

    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "temperature": 0,
        "top_p": 1,
    }

    try:
        result = client.chat.completions.create(**params)
        response_text = result.choices[0].message.content.strip()
        estimated_score = extract_score_from_response(response_text)
        return estimated_score
    except OpenAIError as e:
        print(f"Failed to get response for {file_name}: OpenAI API error - {e}")
        return None
    except ValueError as e:
        print(f"Failed to extract score for {file_name}: ValueError - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {file_name}: {e}")
        return None

# Load examples before scoring the specific video
load_example_dives()

# Process the input video, compare predicted score with actual score
if os.path.isfile(video_file_path):
    print(f"Processing video: {video_file_path}")
    base64_frames = process_video(video_file_path)
    if base64_frames:
        estimated_score = score_dive(video_file_path, base64_frames)

        if estimated_score is not None:
            difference = abs(estimated_score - actual_score)
            print(f"Video: {video_file_path}, Actual Score: {actual_score}, Estimated Score: {estimated_score}, Difference: {difference}")
        else:
            print(f"Failed to score the video: {video_file_path}")
    else:
        print(f"Error: Failed to process the video: {video_file_path}")
else:
    print(f"Error: Video file not found - {video_file_path}")

# Calculate and print total time taken
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")
