import cv2
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load pretrained YOLOv5s model

# Open a connection to the webcam
cap = cv2.VideoCapture(1)  # Change to your webcam index if needed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize occupancy count
occupancy_count = 0
person_ids = set()  # To track unique persons

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference
    results = model(frame)

    # Process results
    detections = results.xyxy[
        0
    ]  # Get detections in the format (x1, y1, x2, y2, conf, class)

    for *box, conf, cls in detections:
        if int(cls) == 0:  # Class ID for "person" in COCO dataset
            x1, y1, x2, y2 = map(int, box)
            # You can add logic here to assign unique IDs for tracking if needed

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Person {occupancy_count}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            occupancy_count += 1  # Increment count for each detected person (you might want to refine this)

    # Display the resulting frame
    cv2.imshow("Occupancy Counting", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
