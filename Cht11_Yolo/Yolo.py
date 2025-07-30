# pip install yolo==0.3.1
# pip install ultralytics

from ultralytics import YOLO #　Installs the library to work with YOLO models.
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

results = model('ImagesAndVideos/image2.jpg')

for result in results:
    count =0
    boxes = result.boxes  # Get the detected box
    print('boxes = ', boxes)  # Print the inspection results

    result.plot()
    plt.imshow(result.plot())
    plt.show()
    plt.savefig('detected_image.png')
    count +=1
print('count = ', count) #1
print('boxes End')

#Save the detected box information directly to the file
# Here you can write into the log
with open('detections.txt', 'w') as f:
    for result in results:
        f.write(str(result.boxes))