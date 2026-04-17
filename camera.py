from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

def create_detectors():
    base_hand_options = BaseOptions(model_asset_path="landmarkers/hand_landmarker.task")
    base_face_options = BaseOptions(model_asset_path="landmarkers/face_landmarker.task")

    face_options = vision.FaceLandmarkerOptions(
        base_options=base_face_options,
        num_faces=1
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=base_hand_options,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )

    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    return hand_detector, face_detector