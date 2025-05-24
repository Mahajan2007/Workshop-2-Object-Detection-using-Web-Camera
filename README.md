# Workshop-2-Object-Detection-using-Web-Camera
## AIM:
To detect the real-life objects with the help of web camera using a program developed in Python whith numpy and cv2.
## ALGORITHM:
1. Load YOLOv4 network with weights and configuration.
2. Load COCO class labels.
3. Get layer names and determine output layers for YOLO.
4. Initialize video capture from webcam.
5. While the webcam is running:
    a. Read the current frame.
    b. Prepare the image by converting it to a blob.
    c. Pass the blob through the YOLOv4 model.
    d. Initialize empty lists for boxes, confidences, and class IDs.
    e. For each output:
        i. Extract detection scores and class IDs.
        ii. If confidence > 0.5, calculate bounding box coordinates.
        iii. Store the box, confidence, and class ID.
    f. Apply Non-Max Suppression to filter overlapping boxes.
    g. Draw bounding boxes and labels for valid detections.
    h. Display the frame with drawn detections.
    i. If the 'q' key is pressed, break the loop.
6. Release webcam and close windows.
## CODE:
```
Developed By:    MAHAJANANI R
Reference Number: 212224230147
```
```
import cv2
import numpy as np
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

       indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

```
## OUTPUT:

![webajith](https://github.com/user-attachments/assets/b01e5ccd-8fe6-493d-941f-f5ad8c45085f)

## RESULT:
Hence,we successfully deployed the code for object detection and real-life objects was successfully detected.
