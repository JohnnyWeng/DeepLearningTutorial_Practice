import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("The camera can't be turned on")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0] # confidence score
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            # (255, 0, 0) -> bgr, 2 -> the wideth of the box line
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) #The width is 2
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow('YOLOv8 Real-time Object Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'): #The lower 8 bits are retained, ignoring irrelevant high data
        break
cap.release()
cv2.destroyAllWindows()
