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
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)  # Perform object detection
    boxes = result[0].boxes
    # Custom visualization with OpenCV
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
        cls = int(box.cls[0])            # Class ID
        conf = box.conf[0]               # Confidence score

        # Draw blue bounding boxes
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue (BGR: 255, 0, 0)

        # Add label text
        label = f"Person {conf:.2f}"  # Update label for person class
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display updated frame
    cv2.imshow("YOLO Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    num_people = len([cls for cls in boxes.cls if cls == 0])  # Count the number of people (class 0 usually represents people)
    print("The number of people in this frame:", num_people)
cap.release()
