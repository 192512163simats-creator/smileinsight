import os
import io
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
# APPLICATION TITLE
# ============================================================

st.title("😊 SmileInsight")

st.subheader(
    "AI-Based Clinical Smile Asymmetry Analysis System"
)

st.info(
    "This application performs AI-assisted facial landmark "
    "analysis and estimates smile asymmetry."
)

st.warning(
    "⚠️ This is a research and educational prototype. "
    "It is NOT a medical diagnosis and should not replace "
    "evaluation by a qualified healthcare professional."
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Face Landmarker model was not found."
    )

    st.write(
        "Your GitHub repository must contain:"
    )

    st.code(
        "models/face_landmarker.task"
    )

    st.write(
        "Your project structure should be:"
    )

    st.code(
        "smileinsight/\n"
        "│\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "│\n"
        "└── models/\n"
        "    └── face_landmarker.task"
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


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:

    landmarker = load_face_landmarker()

except Exception as e:

    st.error(
        "❌ Could not load the MediaPipe Face Landmarker."
    )

    st.error(
        "Check that face_landmarker.task is present "
        "inside the models folder."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CAMERA INPUT
# ============================================================

st.header(
    "📷 Step 1: Capture Your Smile"
)

st.write(
    "Position your face directly in front of your laptop camera."
)

st.write(
    "Keep your head straight and maintain a natural smile."
)

st.write(
    "Click the camera button below."
)


camera_image = st.camera_input(
    "Take a picture of your smile"
)


# ============================================================
# FACE ANALYSIS FUNCTION
# ============================================================

def analyze_face(image):

    # Convert PIL image to RGB
    image_array = np.array(
        image.convert("RGB")
    )

    # Create MediaPipe image
    mp_image = vision.MPImage(
        image_format=vision.ImageFormat.SRGB,
        data=image_array
    )

    # Run face landmark detection
    result = landmarker.detect(
        mp_image
    )

    return image_array, result


# ============================================================
# LANDMARK VISUALIZATION
# ============================================================

def draw_landmarks(
    image_array,
    landmarks
):

    output = image_array.copy()

    height = output.shape[0]
    width = output.shape[1]

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):

            # Draw a small green square
            # without using OpenCV

            x1 = max(
                0,
                x - 2
            )

            x2 = min(
                width,
                x + 3
            )

            y1 = max(
                0,
                y - 2
            )

            y2 = min(
                height,
                y + 3
            )

            output[
                y1:y2,
                x1:x2
            ] = [0, 255, 0]

    return output


# ============================================================
# SMILE ASYMMETRY ANALYSIS
# ============================================================

def calculate_smile_asymmetry(
    landmarks,
    image_width,
    image_height
):

    # --------------------------------------------------------
    # MediaPipe landmark indices
    # --------------------------------------------------------

    LEFT_MOUTH = 61

    RIGHT_MOUTH = 291

    UPPER_LIP = 13

    LOWER_LIP = 14


    # --------------------------------------------------------
    # Extract landmarks
    # --------------------------------------------------------

    left = landmarks[
        LEFT_MOUTH
    ]

    right = landmarks[
        RIGHT_MOUTH
    ]

    upper = landmarks[
        UPPER_LIP
    ]

    lower = landmarks[
        LOWER_LIP
    ]


    # --------------------------------------------------------
    # Convert normalized coordinates
    # to image coordinates
    # --------------------------------------------------------

    left_x = (
        left.x *
        image_width
    )

    left_y = (
        left.y *
        image_height
    )


    right_x = (
        right.x *
        image_width
    )

    right_y = (
        right.y *
        image_height
    )


    upper_x = (
        upper.x *
        image_width
    )

    upper_y = (
        upper.y *
        image_height
    )


    lower_x = (
        lower.x *
        image_width
    )

    lower_y = (
        lower.y *
        image_height
    )


    # --------------------------------------------------------
    # Mouth width
    # --------------------------------------------------------

    mouth_width = np.sqrt(

        (
            right_x -
            left_x
        ) ** 2

        +

        (
            right_y -
            left_y
        ) ** 2

    )


    # --------------------------------------------------------
    # Mouth opening
    # --------------------------------------------------------

    mouth_opening = np.sqrt(

        (
            lower_x -
            upper_x
        ) ** 2

        +

        (
            lower_y -
            upper_y
        ) ** 2

    )


    # --------------------------------------------------------
    # Mouth center
    # --------------------------------------------------------

    mouth_center_x = (

        left_x +

        right_x

    ) / 2


    # --------------------------------------------------------
    # Left and right distances
    # --------------------------------------------------------

    left_distance = abs(

        mouth_center_x -

        left_x

    )


    right_distance = abs(

        right_x -

        mouth_center_x

    )


    # --------------------------------------------------------
    # Calculate asymmetry percentage
    # --------------------------------------------------------

    if mouth_width > 0:

        asymmetry = (

            abs(

                left_distance -

                right_distance

            )

            /

            mouth_width

        ) * 100

    else:

        asymmetry = 0


    # --------------------------------------------------------
    # Limit result
    # --------------------------------------------------------

    asymmetry = max(

        0,

        min(

            asymmetry,

            100

        )

    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if asymmetry < 5:

        classification = (
            "Low asymmetry"
        )

    elif asymmetry < 10:

        classification = (
            "Mild asymmetry"
        )

    elif asymmetry < 20:

        classification = (
            "Moderate asymmetry"
        )

    else:

        classification = (
            "High asymmetry"
        )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "mouth_width":
            mouth_width,

        "mouth_opening":
            mouth_opening,

        "asymmetry":
            asymmetry,

        "classification":
            classification

    }


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def generate_pdf(
    results
):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(

        buffer,

        pagesize=A4

    )

    page_width, page_height = A4


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    pdf.setFont(

        "Helvetica-Bold",

        20

    )

    pdf.drawString(

        50,

        page_height - 60,

        "SmileInsight Analysis Report"

    )


    # --------------------------------------------------------
    # Subtitle
    # --------------------------------------------------------

    pdf.setFont(

        "Helvetica",

        11

    )

    pdf.drawString(

        50,

        page_height - 90,

        "AI-Based Clinical Smile Asymmetry Analysis"

    )


    # --------------------------------------------------------
    # Separator
    # --------------------------------------------------------

    pdf.line(

        50,

        page_height - 105,

        page_width - 50,

        page_height - 105

    )


    # --------------------------------------------------------
    # Analysis Results
    # --------------------------------------------------------

    pdf.setFont(

        "Helvetica-Bold",

        14

    )

    pdf.drawString(

        50,

        page_height - 150,

        "Analysis Results"

    )


    pdf.setFont(

        "Helvetica",

        11

    )


    y = (

        page_height -

        185

    )


    pdf.drawString(

        50,

        y,

        f"Mouth Width: "
        f"{results['mouth_width']:.2f} pixels"

    )


    y -= 25


    pdf.drawString(

        50,

        y,

        f"Mouth Opening: "
        f"{results['mouth_opening']:.2f} pixels"

    )


    y -= 25


    pdf.drawString(

        50,

        y,

        f"Smile Asymmetry: "
        f"{results['asymmetry']:.2f}%"

    )


    y -= 25


    pdf.drawString(

        50,

        y,

        f"Classification: "
        f"{results['classification']}"

    )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    y -= 60


    pdf.setFont(

        "Helvetica-Bold",

        14

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


    pdf.drawString(

        50,

        y,

        "The system detected facial landmarks and "

    )


    y -= 18


    pdf.drawString(

        50,

        y,

        "performed a geometric analysis of selected "

    )


    y -= 18


    pdf.drawString(

        50,

        y,

        "facial mouth landmarks."

    )


    y -= 30


    pdf.drawString(

        50,

        y,

        "This result is intended for research and "

    )


    y -= 18


    pdf.drawString(

        50,

        y,

        "educational purposes only."

    )


    y -= 18


    pdf.drawString(

        50,

        y,

        "It is not a medical diagnosis."

    )


    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    pdf.setFont(

        "Helvetica-Oblique",

        8

    )


    pdf.drawString(

        50,

        60,

        "SmileInsight - AI-Assisted Facial Analysis"

    )


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    pdf.save()

    buffer.seek(0)

    return buffer


# ============================================================
# PROCESS CAMERA IMAGE
# ============================================================

if camera_image is not None:

    st.header(

        "🔍 Step 2: Face Detection"

    )


    try:

        # ----------------------------------------------------
        # Open captured image
        # ----------------------------------------------------

        image = Image.open(

            camera_image

        ).convert(

            "RGB"

        )


        # ----------------------------------------------------
        # Display original image
        # ----------------------------------------------------

        st.image(

            image,

            caption="Captured Smile Image",

            use_container_width=True

        )


        # ----------------------------------------------------
        # Run face analysis
        # ----------------------------------------------------

        image_array, result = analyze_face(

            image

        )


        # ----------------------------------------------------
        # Check if face was detected
        # ----------------------------------------------------

        if not result.face_landmarks:

            st.error(

                "❌ No face detected."

            )

            st.warning(

                "Please position your face directly "
                "in front of the camera and try again."

            )

            st.stop()


        # ----------------------------------------------------
        # Face detected
        # ----------------------------------------------------

        st.success(

            "✅ Face detected successfully!"

        )


        # ----------------------------------------------------
        # Get first face
        # ----------------------------------------------------

        landmarks = (

            result.face_landmarks[0]

        )


        # ----------------------------------------------------
        # Display landmarks
        # ----------------------------------------------------

        landmark_image = draw_landmarks(

            image_array,

            landmarks

        )


        st.subheader(

            "Facial Landmark Detection"

        )


        st.image(

            landmark_image,

            caption="Detected Facial Landmarks",

            use_container_width=True

        )


        # ----------------------------------------------------
        # Get image dimensions
        # ----------------------------------------------------

        image_height = (

            image_array.shape[0]

        )

        image_width = (

            image_array.shape[1]

        )


        # ----------------------------------------------------
        # Calculate asymmetry
        # ----------------------------------------------------

        results = calculate_smile_asymmetry(

            landmarks,

            image_width,

            image_height

        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.header(

            "📊 Step 3: Smile Asymmetry Analysis"

        )


        col1, col2, col3 = st.columns(

            3

        )


        with col1:

            st.metric(

                "Smile Asymmetry",

                f"{results['asymmetry']:.2f}%"

            )


        with col2:

            st.metric(

                "Mouth Width",

                f"{results['mouth_width']:.2f} px"

            )


        with col3:

            st.metric(

                "Mouth Opening",

                f"{results['mouth_opening']:.2f} px"

            )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        st.subheader(

            "Analysis Classification"

        )


        if results["asymmetry"] < 5:

            st.success(

                f"Result: "
                f"{results['classification']}"

            )

        elif results["asymmetry"] < 10:

            st.info(

                f"Result: "
                f"{results['classification']}"

            )

        elif results["asymmetry"] < 20:

            st.warning(

                f"Result: "
                f"{results['classification']}"

            )

        else:

            st.error(

                f"Result: "
                f"{results['classification']}"

            )


        # ----------------------------------------------------
        # Clinical-style interpretation
        # ----------------------------------------------------

        st.subheader(

            "Clinical-Style Interpretation"

        )


        st.write(

            f"The calculated geometric smile "
            f"asymmetry index is approximately "
            f"**{results['asymmetry']:.2f}%**."

        )


        st.write(

            "The result is based on the relative "
            "position of selected facial landmarks "
            "around the mouth."

        )


        st.warning(

            "⚠️ This system is a research prototype "
            "and has not been clinically validated."

        )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.header(

            "📄 Step 4: Generate Report"

        )


        pdf_file = generate_pdf(

            results

        )


        st.download_button(

            label=(
                "⬇️ Download "
                "SmileInsight PDF Report"
            ),

            data=pdf_file,

            file_name=(
                "SmileInsight_"
                "Analysis_Report.pdf"
            ),

            mime="application/pdf"

        )


        st.success(

            "✅ Your SmileInsight report "
            "is ready."

        )


    except Exception as e:

        st.error(

            "❌ An error occurred "
            "during analysis."

        )

        st.exception(e)
