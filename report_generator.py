from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


def generate_report(
    filename,
    patient_id,
    smile_result,
    face_symmetry
):

    # Create PDF document
    document = SimpleDocTemplate(
        filename,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]

    normal_style = styles["BodyText"]

    story = []

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "SMILEINSIGHT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Facial Smile Asymmetry Analysis",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # ---------------------------------------------------------
    # PATIENT INFORMATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Patient / Subject Information",
            heading_style
        )
    )

    patient_data = [
        ["Patient ID", patient_id],
        ["Analysis Type", "Facial Smile Asymmetry"],
    ]

    patient_table = Table(
        patient_data,
        colWidths=[180, 300]
    )

    patient_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    story.append(patient_table)

    story.append(
        Spacer(1, 20)
    )

    # ---------------------------------------------------------
    # SMILE ANALYSIS
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Smile Movement Analysis",
            heading_style
        )
    )

    smile_data = [
        [
            "Measurement",
            "Result"
        ],
        [
            "Left Mouth Corner Movement",
            f"{smile_result['left_movement']:.4f}"
        ],
        [
            "Right Mouth Corner Movement",
            f"{smile_result['right_movement']:.4f}"
        ],
        [
            "Smile Asymmetry Index",
            f"{smile_result['asymmetry_index']:.2f}%"
        ],
        [
            "Geometric Classification",
            smile_result["classification"]
        ]
    ]

    smile_table = Table(
        smile_data,
        colWidths=[300, 180]
    )

    smile_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    story.append(smile_table)

    story.append(
        Spacer(1, 20)
    )

    # ---------------------------------------------------------
    # FACIAL SYMMETRY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Facial Symmetry Analysis",
            heading_style
        )
    )

    symmetry_data = [
        [
            "Parameter",
            "Result"
        ],
        [
            "Geometric Face Symmetry Deviation",
            f"{face_symmetry:.2f}%"
        ]
    ]

    symmetry_table = Table(
        symmetry_data,
        colWidths=[300, 180]
    )

    symmetry_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(symmetry_table)

    story.append(
        Spacer(1, 30)
    )

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Important Clinical Disclaimer",
            heading_style
        )
    )

    disclaimer = (
        "SmileInsight provides an AI-assisted geometric analysis "
        "of facial landmarks captured using a camera. The calculated "
        "measurements are intended for research, educational, and "
        "prototype purposes. This system does not provide a medical "
        "diagnosis and should not be used as a substitute for "
        "professional clinical examination. Any clinical interpretation "
        "must be performed by a qualified healthcare professional."
    )

    story.append(
        Paragraph(
            disclaimer,
            normal_style
        )
    )

    # Build PDF
    document.build(story)