import numpy as np


# ============================================================
# SMILEINSIGHT - FACIAL ASYMMETRY ANALYSIS
# ============================================================


def distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """

    return np.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_smile_asymmetry(
    neutral_landmarks,
    smile_landmarks,
    face_width
):
    """
    Calculate left-right smile asymmetry.

    The input landmarks are normalized coordinates.

    We compare movement of:
    - Left mouth corner
    - Right mouth corner

    Returns:
    - left movement
    - right movement
    - asymmetry index
    - classification
    """

    # MediaPipe mouth corner landmark indices
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # Neutral mouth positions
    neutral_left = neutral_landmarks[LEFT_MOUTH]
    neutral_right = neutral_landmarks[RIGHT_MOUTH]

    # Smile mouth positions
    smile_left = smile_landmarks[LEFT_MOUTH]
    smile_right = smile_landmarks[RIGHT_MOUTH]

    # Calculate movement
    left_movement = distance(
        neutral_left,
        smile_left
    )

    right_movement = distance(
        neutral_right,
        smile_right
    )

    # Prevent division by zero
    average_movement = (
        left_movement +
        right_movement
    ) / 2

    if average_movement == 0:
        asymmetry_index = 0

    else:
        asymmetry_index = (
            abs(left_movement - right_movement)
            / average_movement
        ) * 100

    # Classification
    if asymmetry_index < 5:
        classification = "Minimal asymmetry"

    elif asymmetry_index < 10:
        classification = "Mild asymmetry"

    elif asymmetry_index < 20:
        classification = "Moderate asymmetry"

    else:
        classification = "Marked asymmetry"

    return {
        "left_movement": left_movement,
        "right_movement": right_movement,
        "asymmetry_index": asymmetry_index,
        "classification": classification
    }


def calculate_face_symmetry(landmarks):
    """
    Basic geometric facial symmetry estimate.

    This compares selected facial landmarks
    relative to the nose center.
    """

    # Landmark pairs:
    # left side, right side

    landmark_pairs = [
        (33, 263),     # Eye region
        (133, 362),    # Eye region
        (61, 291),     # Mouth corners
        (70, 300),     # Upper face
        (105, 334)     # Cheek region
    ]

    asymmetry_values = []

    for left_index, right_index in landmark_pairs:

        left_point = landmarks[left_index]
        right_point = landmarks[right_index]

        # Compare horizontal distances
        difference = abs(
            left_point[0] - (1 - right_point[0])
        )

        asymmetry_values.append(difference)

    if len(asymmetry_values) == 0:
        return 0

    average_asymmetry = np.mean(
        asymmetry_values
    )

    # Convert to percentage
    symmetry_score = average_asymmetry * 100

    return symmetry_score