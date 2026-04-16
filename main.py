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

    # convert normalized coords into pixels
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # draw the nodes
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # draw the lines between the nodes
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def is_finger_up(points, tip, pip):
    return points[tip][1] < points[pip][1]


base_options = BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2,
    min_hand_detection_confidence = 0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

counter = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            draw_hand(frame, hand)

            h, w, _ = frame.shape

            points = []
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x,y))

            ## CHECK IF ITS A THUMBS UP GESTURE
            
            ring_above_pinky = points[13][1] < points[17][1] and points[14][1] < points[18][1] and points[15][1] < points[19][1] and points[16][1] < points[20][1]
            middle_above_ring = points[9][1] < points[13][1] and points[10][1] < points[14][1] and points[11][1] < points[15][1] and points[12][1] < points[16][1]
            index_above_middle = points[5][1] < points[9][1] and points[6][1] < points[10][1] and points[7][1] < points[11][1] and points[8][1] < points[12][1]
            thumb_above_index = points[4][1] < points[6][1]

            thumb_extended = points[4][1] < points[3][1] and points[3][1] < points[2][1] and points[2][1] < points[1][1]

            if ring_above_pinky and middle_above_ring and index_above_middle and thumb_above_index and thumb_extended:
                print(f"{counter}. THUMBS UP")
                counter+=1

    cv2.imshow("ESC to exit", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()