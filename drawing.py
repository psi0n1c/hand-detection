import cv2
from config import HAND_CONNECTIONS

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

# ---------------- show image if gesture detected ---------------- #

def show_overlay(show, image):
    if show:
        cv2.imshow("Linganguli", image)
    else:
        try:
            cv2.destroyWindow("Linganguli")
        except:
            pass