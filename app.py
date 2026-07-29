import os
import io
import cv2
import numpy as np
import streamlit as st

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

st.title("SmileInsight")
st.subheader("AI-Assisted Facial Smile Asymmetry Analysis System")

st.warning(
    "This system provides geometric facial analysis for research "
    "and educational purposes. It is not a medical diagnostic system."
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "face_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error("Face Landmarker model not found.")

    st.write("Expected model location:")

    st.code(MODEL_PATH)

    st.info(
        "Please make sure that face_landmarker.task is uploaded "
        "inside a folder named 'models'."
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

    st.error("Could not load the Face Landmarker model.")

    st.code(str(e))

    st.stop()


# ============================================================
# CAMERA INPUT
# ============================================================

st.header("Step 1: Capture Your Smile")

st.write(
    "Position your face directly in front of the camera. "
    "Keep your head straight and smile naturally."
)

camera_image = st.camera_input(
    "Take a picture of your smile"
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_face(image):

    # Convert PIL image to NumPy
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # Convert RGB for MediaPipe
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Convert image to MediaPipe format
    mp_image = vision.MPImage(
        image_format=vision.ImageFormat.SRGB,
        data=image_rgb
    )

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    return image_np, result


# ============================================================
# ANALYSIS
# ============================================================

if camera_image is not None:

    st.header("Step 2: Facial Analysis")

    try:

        # Open captured image
        image = Image.open(
            camera_image
        ).convert("RGB")

        # Analyze face
        image_np, result = analyze_face(
            image
        )

        # Check if face detected
        if not result.face_landmarks:

            st.error(
                "No face detected. "
                "Please take another photo with your face clearly visible."
            )

            st.stop()


        # ====================================================
        # FACE DETECTED
        # ====================================================

        st.success(
            "Face detected successfully!"
        )


        # Get first detected face
        landmarks = result.face_landmarks[0]


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        display_image = image_np.copy()

        height, width, _ = display_image.shape


        # Draw all landmarks
        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                display_image,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


        st.image(
            display_image,
            caption="Detected Facial Landmarks",
            use_container_width=True
        )


        # ====================================================
        # SMILE LANDMARKS
        # ====================================================

        # MediaPipe Face Landmarker mouth landmarks
        # These points are used for geometric analysis.

        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291

        UPPER_LIP = 13
        LOWER_LIP = 14


        # Get coordinates
        left = landmarks[LEFT_MOUTH]
        right = landmarks[RIGHT_MOUTH]

        upper = landmarks[UPPER_LIP]
        lower = landmarks[LOWER_LIP]


        # Convert normalized coordinates
        left_x = left.x * width
        left_y = left.y * height

        right_x = right.x * width
        right_y = right.y * height

        upper_x = upper.x * width
        upper_y = upper.y * height

        lower_x = lower.x * width
        lower_y = lower.y * height


        # ====================================================
        # CALCULATE MOUTH WIDTH
        # ====================================================

        mouth_width = np.sqrt(
            (right_x - left_x) ** 2 +
            (right_y - left_y) ** 2
        )


        # ====================================================
        # CALCULATE MOUTH OPENING
        # ====================================================

        mouth_opening = np.sqrt(
            (lower_x - upper_x) ** 2 +
            (lower_y - upper_y) ** 2
        )


        # ====================================================
        # CALCULATE MOUTH CENTER
        # ====================================================

        mouth_center_x = (
            left_x + right_x
        ) / 2


        # ====================================================
        # LEFT / RIGHT SYMMETRY
        # ====================================================

        left_distance = abs(
            mouth_center_x - left_x
        )

        right_distance = abs(
            right_x - mouth_center_x
        )


        # Prevent division by zero
        if mouth_width > 0:

            asymmetry_percentage = (
                abs(
                    left_distance -
                    right_distance
                )
                / mouth_width
            ) * 100

        else:

            asymmetry_percentage = 0


        # Limit result
        asymmetry_percentage = min(
            asymmetry_percentage,
            100
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if asymmetry_percentage < 5:

            classification = "Low asymmetry"

        elif asymmetry_percentage < 10:

            classification = "Mild asymmetry"

        elif asymmetry_percentage < 20:

            classification = "Moderate asymmetry"

        else:

            classification = "High asymmetry"


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.header(
            "Step 3: Smile Asymmetry Results"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Mouth Width",
                f"{mouth_width:.2f} px"
            )


        with col2:

            st.metric(
                "Mouth Opening",
                f"{mouth_opening:.2f} px"
            )


        with col3:

            st.metric(
                "Asymmetry",
                f"{asymmetry_percentage:.2f}%"
            )


        st.subheader(
            "Analysis Result"
        )


        st.info(
            f"Classification: {classification}"
        )


        # ====================================================
        # SIMPLE INTERPRETATION
        # ====================================================

        st.write(
            "### Interpretation"
        )


        st.write(
            f"The calculated geometric smile asymmetry "
            f"index is approximately "
            f"{asymmetry_percentage:.2f}%."
        )


        st.write(
            "This value represents a geometric comparison "
            "of selected facial landmark positions. "
            "It should not be interpreted as a clinical diagnosis."
        )


        # ====================================================
        # GENERATE PDF REPORT
        # ====================================================

        st.header(
            "Step 4: Generate Report"
        )


        def generate_pdf():

            buffer = io.BytesIO()

            pdf = canvas.Canvas(
                buffer,
                pagesize=A4
            )

            width_a4, height_a4 = A4


            # Title
            pdf.setFont(
                "Helvetica-Bold",
                20
            )

            pdf.drawString(
                50,
                height_a4 - 60,
                "SmileInsight Analysis Report"
            )


            # Subtitle
            pdf.setFont(
                "Helvetica",
                11
            )

            pdf.drawString(
                50,
                height_a4 - 90,
                "AI-Assisted Facial Smile Asymmetry Analysis"
            )


            # Report details
            y = height_a4 - 140


            pdf.setFont(
                "Helvetica-Bold",
                12
            )

            pdf.drawString(
                50,
                y,
                "Analysis Results"
            )


            y -= 30


            pdf.setFont(
                "Helvetica",
                11
            )


            pdf.drawString(
                50,
                y,
                f"Mouth Width: {mouth_width:.2f} pixels"
            )


            y -= 25


            pdf.drawString(
                50,
                y,
                f"Mouth Opening: {mouth_opening:.2f} pixels"
            )


            y -= 25


            pdf.drawString(
                50,
                y,
                f"Asymmetry Index: {asymmetry_percentage:.2f}%"
            )


            y -= 25


            pdf.drawString(
                50,
                y,
                f"Classification: {classification}"
            )


            y -= 50


            pdf.setFont(
                "Helvetica-Bold",
                12
            )


            pdf.drawString(
                50,
                y,
                "Interpretation"
            )


            y -= 30


            pdf.setFont(
                "Helvetica",
                10
            )


            text = pdf.beginText(
                50,
                y
            )


            text.textLine(
                "This report provides geometric facial analysis "
                "based on detected facial landmarks."
            )


            text.textLine(
                "The result is intended for research and educational purposes."
            )


            text.textLine(
                "It is not a medical diagnosis."
            )


            pdf.drawText(
                text
            )


            pdf.save()


            buffer.seek(0)

            return buffer


        pdf_file = generate_pdf()


        st.download_button(
            label="Download Analysis Report",
            data=pdf_file,
            file_name="SmileInsight_Analysis_Report.pdf",
            mime="application/pdf"
        )


    except Exception as e:

        st.error(
            "An error occurred during analysis."
        )

        st.exception(e)
