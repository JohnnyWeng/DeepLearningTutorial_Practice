# pip install yolo==0.3.1
# pip install ultralytics

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2  # For video processing
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = 'ImagesAndVideos/test1.mp4'
# Use opencv
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # First, we use cv2 to separate the video into frames.
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)
    boxes = result[0].boxes  # Get the detected boxes
    num_people = len([cls for cls in boxes.cls if cls == 0])
    print("The number of people in this frame:", num_people)

    plt.imshow(result[0].plot())  # Plot the detection result on the current frame
    plt.axis('off')  # Remove axes for clarity
    plt.show()  # Display the current frame with detections

cap.release()
