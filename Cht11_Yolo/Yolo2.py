# pip install yolo==0.3.1
# pip install ultralytics

from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

results = model('ImagesAndVideos/image3.jpg')

print('results = ', len(results))
for result in results:
    boxes = result.boxes  # Get the detected box

# Visualize inspection results
result.plot()
plt.imshow(result.plot())
plt.show()

plt.savefig('detected_image.png')

print("The number of people: ", len(boxes.cls)) # The number of objects in the image.
