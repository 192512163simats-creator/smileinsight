import cv2
import mediapipe as mp
import numpy as np
import os
import time
import streamlit as st

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from analysis import (
    calculate_smile_asymmetry,
    calculate_face_symmetry
)

from report_generator import generate_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmileInsight",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "SmileInsight"
)

st.subheader(
    "AI-Assisted Facial Smile Asymmetry Analysis System"
)

st.warning(
    "This system provides geometric facial analysis for "
    "research and educational purposes. It is not a medical "
    "diagnostic system."
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    "models/face_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Face Landmarker model not found."
    )

    st.stop()


# ============================================================
# CREATE MEDIAPIPE
# ============================================================

@st.cache_resource
def load_landmarker():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.FaceLandmarker.create_from_options(
        options
    )


landmarker = load_landmarker()


# ============================================================
# SESSION STATE
# ============================================================

if "neutral_landmarks" not in st.session_state:

    st.session_state.neutral_landmarks = None


if "smile_landmarks" not in st.session_state:

    st.session_state.smile_landmarks = None


if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Analysis Controls"
)

patient_id = st.sidebar.text_input(
    "Patient / Subject ID",
    value="SI-001"
)


# ============================================================
# CAMERA INPUT
# ============================================================

st.header(
    "Step 1: Capture Facial Images"
)

st.write(
    "Look directly at the laptop camera."
)

camera_image = st.camera_input(
    "Take a facial image"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if camera_image is not None:

    # Read image bytes
    image_bytes = camera_image.getvalue()

    # Convert bytes to NumPy
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Decode image
    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = landmarker.detect(
        mp_image
    )

    if result.face_landmarks:

        # Get first face
        face_landmarks = result.face_landmarks[0]

        # Convert landmarks to NumPy
        landmarks = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

                for landmark in face_landmarks
            ]
        )

        st.success(
            "Face detected successfully."
        )

        # Display image
        st.image(
            rgb_frame,
            caption="Captured Facial Image"
        )

        # -----------------------------------------------------
        # CAPTURE BUTTONS
        # -----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            if st.button(
                "Capture Neutral Face"
            ):

                st.session_state.neutral_landmarks = (
                    landmarks
                )

                st.success(
                    "Neutral face captured."
                )


        with col2:

            if st.button(
                "Capture Smile"
            ):

                st.session_state.smile_landmarks = (
                    landmarks
                )

                st.success(
                    "Smile captured."
                )


    else:

        st.error(
            "No face detected. Please look directly at the camera."
        )


# ============================================================
# ANALYSIS
# ============================================================

st.header(
    "Step 2: Perform Smile Asymmetry Analysis"
)


if (

    st.session_state.neutral_landmarks is not None

    and

    st.session_state.smile_landmarks is not None

):

    if st.button(
        "Analyze Smile Asymmetry"
    ):

        neutral = (
            st.session_state.neutral_landmarks
        )

        smile = (
            st.session_state.smile_landmarks
        )

        # Estimate face width
        face_width = abs(
            neutral[454][0] -
            neutral[234][0]
        )

        # Calculate smile asymmetry
        smile_result = calculate_smile_asymmetry(
            neutral,
            smile,
            face_width
        )

        # Calculate facial symmetry
        face_symmetry = calculate_face_symmetry(
            neutral
        )

        st.session_state.analysis_result = {

            "smile": smile_result,

            "face_symmetry":
                face_symmetry
        }


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.analysis_result:

    result = (
        st.session_state.analysis_result
    )

    smile = result["smile"]

    face_symmetry = (
        result["face_symmetry"]
    )


    st.header(
        "Step 3: Analysis Results"
    )


    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Left Smile Movement",
            f"{smile['left_movement']:.4f}"
        )


    with col2:

        st.metric(
            "Right Smile Movement",
            f"{smile['right_movement']:.4f}"
        )


    with col3:

        st.metric(
            "Smile Asymmetry",
            f"{smile['asymmetry_index']:.2f}%"
        )


    st.subheader(
        "Geometric Classification"
    )

    st.info(
        smile["classification"]
    )


    # ---------------------------------------------------------
    # FACE SYMMETRY
    # ---------------------------------------------------------

    st.subheader(
        "Facial Symmetry Deviation"
    )

    st.write(
        f"{face_symmetry:.2f}%"
    )


    # ---------------------------------------------------------
    # GENERATE PDF
    # ---------------------------------------------------------

    if st.button(
        "Generate PDF Report"
    ):

        os.makedirs(
            "reports",
            exist_ok=True
        )

        filename = (
            f"reports/"
            f"SmileInsight_{patient_id}.pdf"
        )

        generate_report(

            filename,

            patient_id,

            smile,

            face_symmetry
        )

        st.success(
            "PDF report generated successfully."
        )

        with open(
            filename,
            "rb"
        ) as pdf_file:

            st.download_button(

                label="Download PDF Report",

                data=pdf_file,

                file_name=(
                    f"SmileInsight_{patient_id}.pdf"
                ),

                mime="application/pdf"
            )
