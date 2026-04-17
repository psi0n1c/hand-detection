import cv2
import mediapipe as mp

from camera import create_detectors
from drawing import draw_hand, show_overlay
from gestures import (
    is_thumbs_up,
    is_pointing_up,
    is_fist,
    hand_near_mouth
)

hand_detector, face_detector = create_detectors()

cap = cv2.VideoCapture(0)

thumbsUp_image = cv2.imread("assets/thumbs_up.png")
pointingUp_image = cv2.imread("assets/point_up.png")
fist_image = cv2.imread("assets/fist.jpg")
hand_near_mouth_image = cv2.imread("assets/hand_near_mouth.png")

# ----------- anti-flickering variables ----------- #

thumbs_frames = 0
pointing_frames = 0
fist_frames = 0
hand_near_mouth_frames = 0
showImage = False

# ---------------- MAIN DETECTION LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    hand_result = hand_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    thumbs_up_detected = False
    pointing_up_detected = False
    fist_detected = False
    hand_near_mouth_detected = False

    mouth_center = None

    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]

        h, w, _ = frame.shape

        x = int((face[13].x + face[14].x) / 2 * w)
        y = int((face[13].y + face[14].y) / 2 * h)

        mouth_center = (x, y)

        cv2.circle(frame, mouth_center, 8, (0, 0, 225), -1)

    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name

            draw_hand(frame, hand)

            h, w, _ = frame.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            hand_center = points[9]

            if is_thumbs_up(points, handedness):
                thumbs_up_detected = True

            if is_pointing_up(points):
                pointing_up_detected = True

            if is_fist(points, handedness):
                fist_detected = True

            if mouth_center and hand_near_mouth(hand_center, mouth_center) and is_fist(points, handedness):
                hand_near_mouth_detected = True



    # --------------------------------- #
    # -------- no flickering ---------- #
    # --------------------------------- #

    # ------- THUMBS UP -------- #

    if thumbs_up_detected:
        thumbs_frames += 1
    else:
        thumbs_frames -= 1

    thumbs_frames = max(0, min(thumbs_frames, 10))
    showThumbs = thumbs_frames > 3

    # ------- POINT UP --------- #

    if pointing_up_detected:
        pointing_frames += 1
    else:
        pointing_frames -= 1

    pointing_frames = max(0, min(pointing_frames, 10))
    showPointing = pointing_frames > 3

    # ------- FIST --------- #

    if fist_detected:
        fist_frames += 1
    else:
        fist_frames -= 1

    fist_frames = max(0, min(fist_frames, 10))
    showFist = fist_frames > 3

    # ------- HAND OVER MOUTH --------- #

    if hand_near_mouth_detected:
        hand_near_mouth_frames += 1
    else:
        hand_near_mouth_frames -= 1

    hand_near_mouth_frames = max(0, min(hand_near_mouth_frames, 10))
    showHandNearMouth = hand_near_mouth_frames > 3
        

    if showPointing:
        show_overlay(True, pointingUp_image)
    elif showThumbs:
        show_overlay(True, thumbsUp_image)
    elif showHandNearMouth:
        show_overlay(True, hand_near_mouth_image)
    elif showFist:
        show_overlay(True, fist_image)
    else:
        show_overlay(False, None)

    cv2.imshow("ESC to exit", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()