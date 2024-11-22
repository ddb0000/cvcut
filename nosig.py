import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import concurrent.futures
import torch
import torchvision
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest
import gc

# YOLO Detection (using YOLOv5 with PyTorch)
def detect_objects(frame):
    # Load a pre-trained YOLOv5 model (You can use YOLOv4 or any other depending on your setup)
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load small model for faster inference
    results = model(frame)
    # Perform object detection, filter out unwanted objects
    detected = results.pandas().xywh[0]
    return detected

# Autoencoder to compare frames (simplified)
def load_autoencoder_model():
    # Load a pre-trained Autoencoder model (for example, using Keras)
    return load_model("autoencoder_model.h5")

def compare_frame_to_reference(frame, reference_img, autoencoder):
    # Resize reference image for comparison
    ref_resized = cv2.resize(reference_img, (frame.shape[1], frame.shape[0]))
    
    # Pass both frame and reference image through the Autoencoder to get the reconstruction error
    frame = np.expand_dims(frame, axis=0) / 255.0
    ref_resized = np.expand_dims(ref_resized, axis=0) / 255.0
    
    # Use Autoencoder to reconstruct both the frame and the reference
    reconstructed_frame = autoencoder.predict(frame)
    reconstruction_error = np.mean(np.abs(reconstructed_frame - ref_resized))

    return reconstruction_error

# Frame processing function
def process_frame(frame_data, reference_img, autoencoder, isolation_forest):
    frame_number, frame = frame_data

    # Compare the frame to the reference image (autoencoder-based)
    reconstruction_error = compare_frame_to_reference(frame, reference_img, autoencoder)
    if reconstruction_error < 0.1:  # If the error is low, it is a match
        return None  # Frame is similar to the reference and should be removed

    # Object detection with YOLOv5
    detected_objects = detect_objects(frame)
    if len(detected_objects) > 0:  # Check if specific objects need removal
        return None  # If detection logic triggers, remove the frame

    # Isolation Forest for additional anomaly detection (if required)
    if isolation_forest.predict([frame.flatten()]) == -1:
        return None  # Anomalous frame, remove it

    # If no issues, keep the frame
    return (frame_number, frame)

# Video processing
def process_video(input_file, reference_img, output_file):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output file (compressed video)
    frame_width, frame_height, fps = 640, 360, 15  # Reduce resolution and FPS for fast processing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frames_to_process = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_resized = cv2.resize(frame, (frame_width, frame_height))  # Resize frame for efficiency
        frames_to_process.append((total_frames, frame_resized))  # Store frame number and the actual frame

    # Load Autoencoder model (pre-trained)
    autoencoder = load_autoencoder_model()

    # Initialize Isolation Forest (training it or loading a pre-trained model)
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit([frame.flatten() for _, frame in frames_to_process])

    # Process frames in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_data, reference_img, autoencoder, isolation_forest) 
                   for frame_data in frames_to_process]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Get the result of the processed frame
            if result:  # If not None, keep the frame
                out.write(result[1])

    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_file}")

# Entry point
def main(input_video_path, reference_image_path, output_video_path):
    reference_img = cv2.imread(reference_image_path)
    process_video(input_video_path, reference_img, output_video_path)

# Example usage
input_video_path = "input.mp4"
reference_image_path = "reference.jpg"
output_video_path = "output_filtered.mp4"

main(input_video_path, reference_image_path, output_video_path)
