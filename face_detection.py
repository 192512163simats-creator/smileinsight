import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the MediaPipe Face Landmarker model
MODEL_PATH = "models/face_landmarker.task"

# Create the Face Landmarker options
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create Face Landmarker
landmarker = vision.FaceLandmarker.create_from_options(options)

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Laptop camera not detected")
    exit()

print("SmileInsight Face Landmarker Started")
print("Press Q to quit")

# Frame counter
frame_timestamp_ms = 0

while True:

    # Read camera frame
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read camera")
        break

    # Convert OpenCV BGR image to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect facial landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_timestamp_ms += 33

    # Check whether face was detected
    if result.face_landmarks:

        # Get the first detected face
        face_landmarks = result.face_landmarks[0]

        # Draw every landmark
        for landmark in face_landmarks:

            # Convert normalized coordinates to pixels
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Draw green landmark point
            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        # Display face detected
        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display number of landmarks
        cv2.putText(
            frame,
            f"Landmarks: {len(face_landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        # No face detected
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Display camera
    cv2.imshow(
        "SmileInsight - Face Landmarks",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()

# Close OpenCV windows
cv2.destroyAllWindows()

# Close Face Landmarker
landmarker.close()

print("SmileInsight Face Landmarker Stopped")