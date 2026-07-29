import os
import io
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


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

st.title("😊 SmileInsight")

st.subheader(
    "AI-Assisted Facial Smile Asymmetry Analysis System"
)

st.warning(
    "This system provides geometric facial analysis for "
    "research and educational purposes. It is not a medical "
    "diagnostic system."
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Face Landmarker model not found."
    )

    st.write(
        "Expected model location:"
    )

    st.code(MODEL_PATH)

    st.info(
        "Please place face_landmarker.task inside the "
        "models folder in your GitHub repository."
    )

    st.stop()


# ============================================================
# LOAD MEDIAPIPE FACE LANDMARKER
# ============================================================

@st.cache_resource
def load_face_landmarker():

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

    detector = vision.FaceLandmarker.create_from_options(
        options
    )

    return detector


try:

    landmarker = load_face_landmarker()

except Exception as e:

    st.error(
        "Unable to load the Face Landmarker model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CAMERA INPUT
# ============================================================

st.header("📷 Smile Analysis")

st.write(
    "Allow camera access when your browser asks for permission. "
    "Look directly at the camera and smile naturally."
)

camera_image = st.camera_input(
    "Take a photo using your laptop camera"
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_smile(image_bytes):

    # Convert uploaded image to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Decode image
    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return None, None

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:

        return None, None

    landmarks = result.face_landmarks[0]

    height, width, _ = frame.shape

    # Convert normalized landmarks to pixel coordinates
    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    # ========================================================
    # IMPORTANT FACIAL LANDMARKS
    # ========================================================
    #
    # MediaPipe Face Mesh landmark indexes
    #
    # 61  = Left mouth corner
    # 291 = Right mouth corner
    # 13  = Upper lip center
    # 14  = Lower lip center
    #
    # These are used to estimate smile geometry.
    #

    left_corner = points[61]

    right_corner = points[291]

    upper_lip = points[13]

    lower_lip = points[14]

    # Face center based on mouth corners
    face_center_x = (
        left_corner[0] +
        right_corner[0]
    ) / 2

    # Calculate mouth corner vertical distances
    left_vertical = abs(
        left_corner[1] -
        upper_lip[1]
    )

    right_vertical = abs(
        right_corner[1] -
        upper_lip[1]
    )

    # Avoid division by zero
    denominator = max(
        left_vertical +
        right_vertical,
        1
    )

    # Calculate asymmetry percentage
    asymmetry = (
        abs(
            left_vertical -
            right_vertical
        )
        /
        denominator
    ) * 100

    # Limit to 100
    asymmetry = min(
        asymmetry,
        100
    )

    # Determine dominant side
    if left_vertical > right_vertical:

        dominant_side = "Left side"

    elif right_vertical > left_vertical:

        dominant_side = "Right side"

    else:

        dominant_side = "Balanced"

    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    output_image = frame.copy()

    # Draw mouth landmarks
    cv2.circle(
        output_image,
        left_corner,
        8,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        output_image,
        right_corner,
        8,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        output_image,
        upper_lip,
        8,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        output_image,
        lower_lip,
        8,
        (255, 0, 0),
        -1
    )

    # Draw mouth line
    cv2.line(
        output_image,
        left_corner,
        right_corner,
        (0, 255, 255),
        3
    )

    # Add text
    cv2.putText(
        output_image,
        f"Asymmetry: {asymmetry:.2f}%",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return output_image, {
        "asymmetry": asymmetry,
        "dominant_side": dominant_side,
        "left_vertical": left_vertical,
        "right_vertical": right_vertical
    }


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def generate_pdf(
    asymmetry,
    dominant_side
):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    # Title
    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        height - 60,
        "SmileInsight Analysis Report"
    )

    # Subtitle
    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        height - 90,
        "AI-Assisted Facial Smile Asymmetry Analysis"
    )

    # Line
    pdf.line(
        50,
        height - 105,
        width - 50,
        height - 105
    )

    # Results
    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        height - 150,
        "Analysis Results"
    )

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        height - 185,
        f"Smile Asymmetry Score: {asymmetry:.2f}%"
    )

    pdf.drawString(
        50,
        height - 215,
        f"Dominant Asymmetric Side: {dominant_side}"
    )

    # Interpretation
    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        height - 270,
        "Interpretation"
    )

    pdf.setFont(
        "Helvetica",
        11
    )

    if asymmetry < 5:

        interpretation = (
            "The measured smile geometry appears relatively balanced."
        )

    elif asymmetry < 15:

        interpretation = (
            "A mild geometric difference between the two sides "
            "of the smile was observed."
        )

    else:

        interpretation = (
            "A noticeable geometric difference between the two "
            "sides of the smile was observed."
        )

    pdf.drawString(
        50,
        height - 300,
        interpretation
    )

    # Disclaimer
    pdf.setFont(
        "Helvetica-Oblique",
        9
    )

    pdf.drawString(
        50,
        100,
        "This report is intended for research and educational purposes."
    )

    pdf.drawString(
        50,
        85,
        "It is not a medical diagnosis or clinical assessment."
    )

    pdf.save()

    buffer.seek(0)

    return buffer


# ============================================================
# PROCESS CAMERA IMAGE
# ============================================================

if camera_image is not None:

    st.info(
        "Analyzing your facial landmarks..."
    )

    try:

        result_image, analysis = analyze_smile(
            camera_image.getvalue()
        )

        # ====================================================
        # FACE NOT DETECTED
        # ====================================================

        if result_image is None:

            st.error(
                "No face detected. Please take another photo."
            )

        else:

            # Convert BGR to RGB for Streamlit
            display_image = cv2.cvtColor(
                result_image,
                cv2.COLOR_BGR2RGB
            )

            # =================================================
            # DISPLAY ANALYZED IMAGE
            # =================================================

            st.subheader(
                "Face Landmark Analysis"
            )

            st.image(
                display_image,
                caption="Detected facial landmarks",
                use_container_width=True
            )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.subheader(
                "📊 Smile Asymmetry Results"
            )

            asymmetry = analysis[
                "asymmetry"
            ]

            dominant_side = analysis[
                "dominant_side"
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Smile Asymmetry",
                    f"{asymmetry:.2f}%"
                )

            with col2:

                st.metric(
                    "Dominant Side",
                    dominant_side
                )

            # =================================================
            # SIMPLE INTERPRETATION
            # =================================================

            if asymmetry < 5:

                st.success(
                    "The measured smile geometry appears relatively balanced."
                )

            elif asymmetry < 15:

                st.warning(
                    "A mild geometric difference was detected "
                    "between the two sides of the smile."
                )

            else:

                st.error(
                    "A noticeable geometric difference was detected. "
                    "This result is not a medical diagnosis."
                )

            # =================================================
            # GENERATE REPORT
            # =================================================

            st.subheader(
                "📄 Generate Report"
            )

            pdf_file = generate_pdf(
                asymmetry,
                dominant_side
            )

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_file,
                file_name="SmileInsight_Report.pdf",
                mime="application/pdf"
            )
