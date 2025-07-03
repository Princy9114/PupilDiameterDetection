import cv2
import numpy as np
import time
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

# Load your trained models
reg_model = joblib.load(r"D:\Princy\Internship\CSIR-CRRI\pupil\final\regression_model.pkl")
clf_model = joblib.load(r"D:\Princy\Internship\CSIR-CRRI\pupil\final\classification_model.pkl")
label_encoder = joblib.load(r"D:\Princy\Internship\CSIR-CRRI\pupil\final\label_encoder.pkl")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"D:\Princy\Internship\CSIR-CRRI\pupil\final\face_landmarker_v2_with_blendshapes.task"),
    running_mode=VisionRunningMode.VIDEO
)

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp)

        if results.face_landmarks and len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]

            if len(landmarks) > 473:
                # Get pupil positions
                left_pupil = landmarks[468]
                right_pupil = landmarks[473]

                # Convert to pixel coordinates
                left_x, left_y = int(left_pupil.x * frame.shape[1]), int(left_pupil.y * frame.shape[0])
                right_x, right_y = int(right_pupil.x * frame.shape[1]), int(right_pupil.y * frame.shape[0])

                # Calculate diameter
                diameter = np.linalg.norm([left_x - right_x, left_y - right_y])

                # Generate dummy values for alignment, light, and time
                Alignment = np.random.choice([0, 1, 2])
                light = np.random.choice([0, 1])
                time_sec = round(time.time() % 1000, 2)

                # Prediction
                reg_input = pd.DataFrame([[time_sec, Alignment, light]], columns=['time_sec', 'Alignment', 'light'])
                class_input = pd.DataFrame([[time_sec, Alignment, light, diameter, diameter]], columns=['time_sec', 'Alignment', 'light', 'Pupil diameter left', 'Pupil diameter right'])

                predicted_diameter = reg_model.predict(reg_input)[0]
                predicted_class = clf_model.predict(class_input)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                # Display info
                cv2.putText(frame, f"Diameter: {diameter:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted: {predicted_label}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Draw circles
                cv2.circle(frame, (left_x, left_y), 3, (255, 0, 0), -1)
                cv2.circle(frame, (right_x, right_y), 3, (0, 0, 255), -1)
            else:
                print("⚠️ Not enough landmarks detected.")
        else:
            print("⚠️ No face or landmarks found.")

        cv2.imshow("Pupil Diameter Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()