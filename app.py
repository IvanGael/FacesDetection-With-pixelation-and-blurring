

import cv2
import torch

# Load YOLOv5 model from ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set the model to detect only persons
model.classes = [0]  # Class 0 corresponds to 'person'

# Open the webcam
cap = cv2.VideoCapture("video.mp4")

def blur_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face = image[y1:y2, x1:x2]
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    image[y1:y2, x1:x2] = blurred_face
    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection
    results = model(frame)

    # Extract bounding boxes for detected persons
    person_bboxes = results.xyxy[0].cpu().numpy()

    for bbox in person_bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        if cls == 0 and conf > 0.5:  # Only process if it's a person and confidence is high
            person_region = frame[int(y1):int(y2), int(x1):int(x2)]

            # Detect faces within the person region
            faces = face_cascade.detectMultiScale(person_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                face_bbox = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
                frame = blur_face(frame, face_bbox)

    # Display the frame with blurred faces
    cv2.imshow('Face Blur', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
