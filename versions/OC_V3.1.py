import cv2
import torch
import warnings
import numpy as np
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Open a connection to the webcam
cap = cv2.VideoCapture(1)  # Change to your webcam index if needed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize person boxes
person_boxes = []  # List to store currently tracked person boxes
detection_history = deque(maxlen=5)  # Queue to stabilize detections


# Function to calculate intersection over union
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


# Check if a new bounding box is similar to an existing one
def is_new_person(new_box, existing_boxes, threshold=0.5):
    for box in existing_boxes:
        if iou(new_box, box) > threshold:
            return False
    return True


# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0]

    # Current detections for this frame
    detected_boxes = []
    for *box, conf, cls in detections:
        if int(cls) == 0 and conf > 0.5:
            box = list(map(int, box))
            detected_boxes.append(box)

    # Add to history and stabilize by averaging recent frames
    detection_history.append(detected_boxes)
    stable_boxes = []
    for box in detected_boxes:
        if is_new_person(box, stable_boxes):
            stable_boxes.append(box)

    # Update person_boxes and track using IOU to prevent flickering
    person_boxes = []
    for box in stable_boxes:
        if is_new_person(box, person_boxes):
            person_boxes.append(box)

    # Draw bounding boxes for tracked persons
    for box in person_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Set occupancy count to the number of current bounding boxes
    occupancy_count = len(person_boxes)

    # Display occupancy count
    cv2.putText(
        frame,
        f"Occupancy Count: {occupancy_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow("Occupancy Counting", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
