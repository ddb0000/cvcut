import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Define the range of "blue" color
def is_blue_color(pixel):
    # Check if blue is significantly higher than green and red
    blue, green, red = pixel
    return blue > red and blue > green

# Threshold to consider a frame as "solid blue"
threshold = 0.95  # Percentage of blue pixels in the frame (tune this value)

def is_solid_blue_frame(frame):
    # Count the number of blue pixels
    blue_pixels = np.sum([is_blue_color(pixel) for pixel in frame.reshape(-1, 3)])
    total_pixels = frame.shape[0] * frame.shape[1]  # Total number of pixels in the frame
    blue_ratio = blue_pixels / total_pixels  # Ratio of blue pixels

    print(f"Blue pixels: {blue_pixels}, Total pixels: {total_pixels}, Blue ratio: {blue_ratio:.4f}")  # Debug print

    return blue_ratio > threshold  # Frame is considered solid blue if > threshold of pixels match blue

def process_frame(frame, frame_number, blue_frames_found):
    # Check if the frame is solid blue
    if is_solid_blue_frame(frame):
        print(f"Solid blue frame detected: Frame {frame_number}")  # Debug print
        blue_frames_found += 1
        return None  # Blue frame, we skip it
    else:
        print(f"Frame {frame_number} processed.")  # Debug print
        return frame, blue_frames_found  # Non-blue frame, we keep it

def process_video(input_file, output_file):
    # Open the input video
    cap = cv2.VideoCapture(input_file)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Load the audio from the original video
    video = VideoFileClip(input_file)
    audio = video.audio

    # Get video properties
    fps = video.fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: FPS = {fps}, Width = {frame_width}, Height = {frame_height}")  # Debug print

    # Set up the VideoWriter for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (frame_width, frame_height))

    total_frames = 0
    blue_frames_found = 0

    # Use ThreadPoolExecutor for parallel processing, but process frames one at a time to avoid memory overload
    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or error in reading a frame.")
                break
            total_frames += 1

            # Resize frames to reduce memory usage (e.g., downscale by 50%)
            frame_resized = cv2.resize(frame, (frame_width // 2, frame_height // 2))

            # Submit frame for processing
            futures.append(executor.submit(process_frame, frame_resized, total_frames, blue_frames_found))

        # Collect results as they are completed
        for future in as_completed(futures):
            result, blue_frames_found = future.result()
            if result is not None:  # If it's not a blue frame, write it
                out.write(result)

            # Force garbage collection to free memory
            gc.collect()

    cap.release()
    out.release()
    video.close()

    # Summary of detected frames
    print(f"Total frames processed: {total_frames}")
    print(f"Total blue frames found: {blue_frames_found}")

    # Reload the temp video for audio synchronization
    new_video = VideoFileClip('temp_video.mp4')

    # Calculate the new audio duration based on saved frames
    audio_duration = (total_frames - blue_frames_found) / fps  # New duration
    new_audio = audio.subclip(0, audio_duration)  # Keep only the relevant audio

    # Set the audio of the new video to the trimmed audio
    final_video = new_video.set_audio(new_audio)
    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac')

    # Cleanup temporary file
    new_video.close()
    final_video.close()

# Define input and output video paths
input_video_path = 'input.mp4'
output_video_path = 'output_no_signal.mp4'

# Process the video
process_video(input_video_path, output_video_path)