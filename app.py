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
    "AI-Assisted Facial Smile Asymmetry Analysis System"
)

st.warning(
    "This application provides geometric facial analysis "
    "for research and educational purposes. It is not a "
    "medical diagnosis."
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
# CHECK FACE LANDMARKER MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Face Landmarker model not found."
    )

    st.write(
        "The application expects the model at:"
    )

    st.code(
        "models/face_landmarker.task"
    )

    st.write(
        "Please make sure your GitHub repository has this structure:"
    )

    st.code(
        "models/\n"
        "└── face_landmarker.task"
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
# INITIALIZE LANDMARKER
# ============================================================

try:

    landmarker = load_face_landmarker()

except Exception as e:

    st.error(
        "❌ Could not load the Face Landmarker."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CAMERA SECTION
# ============================================================

st.header(
    "📷 Step 1: Capture Your Smile"
)

st.write(
    "Look directly at your laptop camera."
)

st.write(
    "Keep your head straight and smile naturally."
)

st.write(
    "Click the camera button below and allow camera permission."
)


camera_image = st.camera_input(
    "Take a picture of your smile"
)


# ============================================================
# FACE ANALYSIS FUNCTION
# ============================================================

def analyze_face(image):

    # Convert PIL image to NumPy array
    image_array = np.array(
        image
    )

    # Ensure RGB format
    if image_array.shape[-1] == 4:

        image_array = image_array[:, :, :3]

    # Create MediaPipe image
    mp_image = vision.MPImage(
        image_format=vision.ImageFormat.SRGB,
        data=image_array
    )

    # Detect facial landmarks
    result = landmarker.detect(
        mp_image
    )

    return image_array, result


# ============================================================
# DRAW FACIAL LANDMARKS
# ============================================================

def draw_landmarks(
    image_array,
    landmarks
):

    # Make a copy
    output = image_array.copy()

    height = output.shape[0]
    width = output.shape[1]

    # Draw all face landmarks
    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        # Check valid coordinates
        if (
            0 <= x < width
            and
            0 <= y < height
        ):

            # Draw a small point
            output[
                max(0, y - 2):min(height, y + 3),
                max(0, x - 2):min(width, x + 3)
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
    # MediaPipe Face Landmark Indices
    # --------------------------------------------------------
    #
    # 61  = Left mouth corner
    # 291 = Right mouth corner
    # 13  = Upper lip center
    # 14  = Lower lip center
    #
    # These landmarks are used for basic geometric analysis.
    #

    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    UPPER_LIP = 13
    LOWER_LIP = 14


    # --------------------------------------------------------
    # Get landmarks
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
    # Convert normalized coordinates to pixels
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
    # Calculate mouth width
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
    # Calculate mouth opening
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
    # Calculate mouth center
    # --------------------------------------------------------

    mouth_center_x = (
        left_x +
        right_x
    ) / 2


    # --------------------------------------------------------
    # Calculate left and right distances
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
    # Calculate asymmetry
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


    # Keep value between 0 and 100
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
            classification,

        "left_x":
            left_x,

        "left_y":
            left_y,

        "right_x":
            right_x,

        "right_y":
            right_y,

        "upper_x":
            upper_x,

        "upper_y":
            upper_y,

        "lower_x":
            lower_x,

        "lower_y":
            lower_y
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
        "AI-Assisted Facial Smile Asymmetry Analysis"
    )


    # --------------------------------------------------------
    # Line
    # --------------------------------------------------------

    pdf.line(
        50,
        page_height - 105,
        page_width - 50,
        page_height - 105
    )


    # --------------------------------------------------------
    # Results heading
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


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    pdf.setFont(
        "Helvetica",
        11
    )


    y = page_height - 185


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
        "performed a geometric comparison"
    )


    y -= 18


    pdf.drawString(
        50,
        y,
        "of selected left and right mouth positions."
    )


    y -= 18


    pdf.drawString(
        50,
        y,
        "The result is intended for research and "
        "educational purposes only."
    )


    y -= 18


    pdf.drawString(
        50,
        y,
        "It should not be considered a medical diagnosis."
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


    pdf.save()


    buffer.seek(0)


    return buffer


# ============================================================
# PROCESS CAMERA IMAGE
# ============================================================

if camera_image is not None:

    st.header(
        "🔍 Step 2: Detecting Face"
    )


    try:

        # Open captured image
        image = Image.open(
            camera_image
        ).convert("RGB")


        # Display original image
        st.image(
            image,
            caption="Captured Image",
            use_container_width=True
        )


        # Analyze face
        image_array, result = analyze_face(
            image
        )


        # ----------------------------------------------------
        # Check face detection
        # ----------------------------------------------------

        if not result.face_landmarks:

            st.error(
                "❌ No face detected."
            )

            st.warning(
                "Please make sure your face is clearly visible, "
                "look directly at the camera, and try again."
            )

            st.stop()


        # ----------------------------------------------------
        # Face detected
        # ----------------------------------------------------

        st.success(
            "✅ Face detected successfully!"
        )


        # Get first face
        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # Draw landmarks
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
        # Calculate smile asymmetry
        # ----------------------------------------------------

        image_height = (
            image_array.shape[0]
        )

        image_width = (
            image_array.shape[1]
        )


        results = calculate_smile_asymmetry(
            landmarks,
            image_width,
            image_height
        )


        # ====================================================
        # DISPLAY ANALYSIS
        # ====================================================

        st.header(
            "📊 Step 3: Smile Asymmetry Analysis"
        )


        col1, col2, col3 = st.columns(3)


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
                f"Result: {results['classification']}"
            )

        elif results["asymmetry"] < 10:

            st.info(
                f"Result: {results['classification']}"
            )

        elif results["asymmetry"] < 20:

            st.warning(
                f"Result: {results['classification']}"
            )

        else:

            st.error(
                f"Result: {results['classification']}"
            )


        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        st.subheader(
            "Clinical-Style Interpretation"
        )


        st.write(
            f"The calculated geometric smile asymmetry "
            f"index is approximately "
            f"**{results['asymmetry']:.2f}%**."
        )


        st.write(
            "This measurement is based on the geometric "
            "positions of selected facial landmarks."
        )


        st.warning(
            "⚠️ This is a research prototype and not a "
            "clinically validated medical diagnostic tool."
        )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.header(
            "📄 Step 4: Generate Analysis Report"
        )


        pdf_file = generate_pdf(
            results
        )


        st.download_button(
            label="⬇️ Download SmileInsight PDF Report",
            data=pdf_file,
            file_name="SmileInsight_Analysis_Report.pdf",
            mime="application/pdf"
        )


        st.success(
            "✅ Your analysis report is ready."
        )


    except Exception as e:

        st.error(
            "❌ An error occurred while analyzing the image."
        )

        st.exception(e)
