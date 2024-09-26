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
```Python
import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

```
## OUTPUT:
![Screenshot 2024-09-26 161130](https://github.com/user-attachments/assets/f31652c6-7863-48a7-9739-c62196293c43)

## RESULT:
Hence,we successfully deployed the code for object detection and real-life objects was successfully detected.
