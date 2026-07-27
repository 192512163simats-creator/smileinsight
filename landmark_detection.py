import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# STEP 1: LOCATION OF THE AI MODEL
# ==================================================

MODEL_PATH = "models/face_landmarker.task"


# ==================================================
# STEP 2: CREATE MEDIAPIPE FACE LANDMARKER
# ==================================================

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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ==================================================
# STEP 3: OPEN YOUR LAPTOP CAMERA
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Laptop camera not detected")
    exit()

print("SmileInsight started")
print("Laptop camera is running")
print("Press Q to close")


# ==================================================
# STEP 4: FRAME TIMESTAMP
# ==================================================

frame_timestamp_ms = 0


# ==================================================
# STEP 5: START CAMERA LOOP
# ==================================================

while True:

    # Read a frame from laptop camera
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read camera")
        break


    # ==================================================
    # STEP 6: CONVERT BGR TO RGB
    # ==================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ==================================================
    # STEP 7: CONVERT IMAGE TO MEDIAPIPE FORMAT
    # ==================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ==================================================
    # STEP 8: DETECT FACE LANDMARKS
    # ==================================================

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_timestamp_ms += 33


    # ==================================================
    # STEP 9: CHECK IF FACE IS DETECTED
    # ==================================================

    if result.face_landmarks:

        # Get the first detected face
        face_landmarks = result.face_landmarks[0]


        # ==================================================
        # STEP 10: DRAW FACIAL LANDMARK POINTS
        # ==================================================

        for landmark in face_landmarks:

            # Convert normalized coordinates to pixels
            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            # Draw green point
            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )


        # ==================================================
        # STEP 11: DISPLAY FACE DETECTED
        # ==================================================

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

        # ==================================================
        # NO FACE DETECTED
        # ==================================================

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==================================================
    # STEP 12: SHOW CAMERA WINDOW
    # ==================================================

    cv2.imshow(
        "SmileInsight - Facial Landmarks",
        frame
    )


    # ==================================================
    # STEP 13: PRESS Q TO EXIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==================================================
# STEP 14: CLOSE CAMERA AND PROGRAM
# ==================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()

print("SmileInsight stopped")