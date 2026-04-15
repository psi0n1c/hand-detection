import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),# ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)
]

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape

    points = []

    # convert normalized coords → pixels
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # draw points
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


counter = 0

base_options = BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2,
    min_hand_detection_confidence = 0.01
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        print(f"{counter}. Hand detected")
        counter += 1
        for hand in result.hand_landmarks:
            draw_hand(frame, hand)

    cv2.imshow("ESC to exit", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()