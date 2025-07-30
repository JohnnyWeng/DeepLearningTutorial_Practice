# pip install yolo==0.3.1
# pip install ultralytics

from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
model = YOLO('yolov8n.pt')
print('model.names = ', model.names)

results = model('ImagesAndVideos/image4.jpg')

car_count = 0
motorbike_count = 0
person_count = 0
print('result = ', len(results))
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        # Map class ID to label
        label = model.names[class_id]
        print('label = ', label)
        # Count cats and dogs
        if label == "car":
            car_count += 1
        if label == 'person':
            person_count += 1
        elif label == "motorcycle":
            motorbike_count += 1
    print('result.boxes = ', len(result.boxes))
# Visualize inspection results
result.plot()
plt.imshow(result.plot())
plt.show()

# To save the visualization to a file, you can use matplotlib
plt.savefig('detected_image.png')

print(f"Number of cars: {car_count}")
print(f"Number of person: {person_count}")
print(f"Number of motorcycle: {motorbike_count}")
