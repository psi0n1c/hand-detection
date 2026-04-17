import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------------- HAND SKELETON NODES ---------------- #

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # index
    (5, 9), (9, 10), (10, 11), (11, 12), # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17)
]

# ---------------- draw hand skeletons ---------------- #

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # draw nodes
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # draw lines connecting nodes
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

# ---------------- GESTURE: THUMBS UP ---------------- #

def is_thumbs_up(points, handedness):
    ring_above_pinky = (
        points[13][1] < points[17][1] and
        points[14][1] < points[18][1] and
        points[15][1] < points[19][1] and
        points[16][1] < points[20][1]
    )

    middle_above_ring = (
        points[9][1] < points[13][1] and
        points[10][1] < points[14][1] and
        points[11][1] < points[15][1] and
        points[12][1] < points[16][1]
    )

    index_above_middle = (
        points[5][1] < points[9][1] and
        points[6][1] < points[10][1] and
        points[7][1] < points[11][1] and
        points[8][1] < points[12][1]
    )

    thumb_above_index = points[4][1] < points[6][1]

    all_above_eachother = (
        ring_above_pinky and
        middle_above_ring and
        index_above_middle and
        thumb_above_index
    )

    thumb_extended = (
        points[4][1] < points[3][1] <
        points[2][1] < points[1][1]
    )

    if handedness == "Right":
        pinky_tucked = points[20][0] < points[18][0]
        ring_tucked = points[16][0] < points[14][0]
        middle_tucked = points[12][0] < points[10][0]
        index_tucked = points[8][0] < points[6][0]
        thumb_upright = points[4][0] < points[5][0]
    else:
        pinky_tucked = points[20][0] > points[18][0]
        ring_tucked = points[16][0] > points[14][0]
        middle_tucked = points[12][0] > points[10][0]
        index_tucked = points[8][0] > points[6][0]
        thumb_upright = points[4][0] > points[5][0]

    all_tucked = (
        pinky_tucked and
        ring_tucked and
        middle_tucked and
        index_tucked
    )

    return all_tucked and all_above_eachother and thumb_extended and thumb_upright

# ---------------- GESTURE: POINT UP LIKE A NERD ---------------- #

def is_pointing_up(points):
    index_up = points[8][1] < points[6][1]

    middle_down = points[12][1] > points[10][1]
    ring_down   = points[16][1] > points[14][1]
    pinky_down  = points[20][1] > points[18][1]

    return index_up and middle_down and ring_down and pinky_down

# ---------------- show image if gesture detected ---------------- #

def show_overlay(show, image):
    if show:
        cv2.imshow("Linganguli", image)
    else:
        try:
            cv2.destroyWindow("Linganguli")
        except:
            pass

# ---------------- camera detection setup ---------------- #

base_options = BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

thumbsUp_image = cv2.imread("assets/thumbs_up.png")
pointingUp_image = cv2.imread("assets/point_up.png")

# ----------- anti-flickering variables ----------- #

thumbs_frames = 0
pointing_frames = 0
showImage = False

# ---------------- MAIN DETECTION LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    thumbs_up_detected = False
    pointing_up_detected = False

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name

            draw_hand(frame, hand)

            h, w, _ = frame.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            if is_thumbs_up(points, handedness):
                thumbs_up_detected = True

            if is_pointing_up(points):
                pointing_up_detected = True

    # -------- no flickering -------- #

    if thumbs_up_detected:
        thumbs_frames += 1
    else:
        thumbs_frames -= 1

    thumbs_frames = max(0, min(thumbs_frames, 10))
    showThumbs = thumbs_frames > 3

    if pointing_up_detected:
        pointing_frames += 1
    else:
        pointing_frames -= 1

    pointing_frames = max(0, min(pointing_frames, 10))
    showPointing = pointing_frames > 3

    if showPointing:
        show_overlay(True, pointingUp_image)
    elif showThumbs:
        show_overlay(True, thumbsUp_image)
    else:
        show_overlay(False, None)

    cv2.imshow("ESC to exit", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()