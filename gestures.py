import math

# ------- FINGERS ARE TUCKED WHEN HAND IS SIDEWAYS -------------- #

def are_tucked_sideways(points, handedness):
    if handedness == "Right":
        pinky_tucked = points[20][0] < points[18][0]
        ring_tucked = points[16][0] < points[14][0]
        middle_tucked = points[12][0] < points[10][0]
        index_tucked = points[8][0] < points[6][0]

    else:
        pinky_tucked = points[20][0] > points[18][0]
        ring_tucked = points[16][0] > points[14][0]
        middle_tucked = points[12][0] > points[10][0]
        index_tucked = points[8][0] > points[6][0]

    all_tucked = (
        pinky_tucked and
        ring_tucked and
        middle_tucked and
        index_tucked
    )

    return all_tucked

# ------- HAND IS NEAR MOUTH -------------- #

def hand_near_mouth(hand_center, mouth_center, threshold=80):
    if mouth_center == None:
        return False
    
    dx = hand_center[0] - mouth_center[0]
    dy = hand_center[1] - mouth_center[1]

    distance = math.sqrt(dx*dx + dy*dy)

    return distance < threshold

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
        thumb_upright = points[4][0] < points[5][0]
    else:
        thumb_upright = points[4][0] > points[5][0]
    
    return are_tucked_sideways(points, handedness) and all_above_eachother and thumb_extended and thumb_upright

# ---------------- GESTURE: POINT UP LIKE A NERD ---------------- #

def is_pointing_up(points):
    index_up = points[8][1] < points[6][1]

    middle_down = points[12][1] > points[10][1]
    ring_down   = points[16][1] > points[14][1]
    pinky_down  = points[20][1] > points[18][1]

    return index_up and middle_down and ring_down and pinky_down

# ---------------- GESTURE: FIST ANGRY ---------------- #

def is_fist(points, handedness):
    return are_tucked_sideways(points, handedness)
