import cv2
import torch
import warnings
import numpy as np

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
detected_persons = {}  # Dictionary to hold the unique IDs of persons

# Constants
DISTANCE_THRESHOLD = 50  # Distance threshold to match detected persons
FRAMES_LOST_THRESHOLD = (
    50  # Number of frames to keep a person "alive" after last detection
)

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

    current_person_ids = []  # List to track IDs for the current frame

    for *box, conf, cls in detections:
        if int(cls) == 0:  # Class ID for "person" in COCO dataset
            x1, y1, x2, y2 = map(int, box)

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if the person is already tracked
            found = False
            for person_id, (last_center, frames_lost) in detected_persons.items():
                last_center_x, last_center_y = last_center
                if (
                    np.linalg.norm([center_x - last_center_x, center_y - last_center_y])
                    < DISTANCE_THRESHOLD
                ):
                    # Update position and reset frames lost
                    detected_persons[person_id] = (
                        (center_x, center_y),
                        0,
                    )  # Reset frames lost
                    current_person_ids.append(person_id)  # Track this ID
                    found = True
                    break

            if not found:  # If not found, it's a new person
                occupancy_count += 1
                person_id = occupancy_count
                detected_persons[person_id] = (
                    (center_x, center_y),
                    0,
                )  # Add new person ID with center position
                current_person_ids.append(person_id)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Person {person_id}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Update detected persons with the current positions and track frames lost
    new_detected_persons = {}
    for person_id in current_person_ids:
        if person_id in detected_persons:
            new_detected_persons[person_id] = detected_persons[person_id]

    # Check for persons that are not detected in the current frame
    for person_id in detected_persons:
        if person_id not in current_person_ids:
            last_center, frames_lost = detected_persons[person_id]
            frames_lost += 1
            if (
                frames_lost < FRAMES_LOST_THRESHOLD
            ):  # Keep the person "alive" for a few frames
                new_detected_persons[person_id] = (
                    last_center,
                    frames_lost,
                )  # Update with frames lost

    detected_persons = new_detected_persons

    # Display the resulting frame
    cv2.imshow("Occupancy Counting", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
