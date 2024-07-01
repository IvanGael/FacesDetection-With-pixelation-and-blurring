import cv2
import torch
import mediapipe as mp
import numpy as np
import argparse

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]  # Only detect persons (class 0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.5)

def pixelate_face(image, landmarks, blocks=15):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(landmarks)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    x, y, w, h = cv2.boundingRect(hull)
    face = image[y:y+h, x:x+w]
    face_mask = mask[y:y+h, x:x+w]

    temp = cv2.resize(face, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    result = image.copy()
    result[y:y+h, x:x+w] = np.where(face_mask[:, :, None] == 255, pixelated_face, face)

    return result

def blur_face(image, landmarks, blur_factor=99):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(landmarks)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    x, y, w, h = cv2.boundingRect(hull)
    face = image[y:y+h, x:x+w]
    face_mask = mask[y:y+h, x:x+w]

    blurred_face = cv2.GaussianBlur(face, (blur_factor, blur_factor), 0)

    result = image.copy()
    result[y:y+h, x:x+w] = np.where(face_mask[:, :, None] == 255, blurred_face, face)

    return result


def process_video(input_video, output_video, method='pixelation'):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:  # Infinite loop to keep restarting the video
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning of the video
            continue

        frame_count += 1
        
        # Detect persons using YOLOv5
        results = model(frame)
        person_boxes = results.xyxy[0].cpu().numpy()
        
        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces_landmarks = []
        person_count = 0
        for box in person_boxes:
            if box[4] > 0.5:  # Confidence threshold
                person_count += 1
                x1, y1, x2, y2 = map(int, box[:4])
                person_region = rgb_frame[y1:y2, x1:x2]
                
                # Process the person region to find faces
                face_results = face_mesh.process(person_region)
                
                if face_results.multi_face_landmarks:
                    h, w, _ = person_region.shape
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Convert landmarks to pixel coordinates relative to the whole frame
                        landmarks = [(int(point.x * w + x1), int(point.y * h + y1)) for point in face_landmarks.landmark]
                        faces_landmarks.append(landmarks)

        # Apply chosen method to all detected faces
        for landmarks in faces_landmarks:
            if method == 'pixelation':
                frame = pixelate_face(frame, landmarks)
            elif method == 'blurring':
                frame = blur_face(frame, landmarks)

        # Display debugging information
        cv2.putText(frame, f'Frame: {frame_count}, Faces: {person_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame to the output video file
        out.write(frame)

        # Display the frame with processed faces
        cv2.imshow('Face Processing', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Restart video when frame count reaches 90
        if frame_count == 90:
            frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning of the video

    # Release the capture and close the windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face pixelation or blurring in video")
    parser.add_argument("--method", choices=['pixelation', 'blurring'], default='pixelation',
                        help="Choose between 'pixelation' and 'blurring' (default: pixelation)")
    args = parser.parse_args()

    process_video('video.mp4', 'output.avi', args.method)

face_mesh.close()