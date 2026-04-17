# Hand Gesture Recognition System

A real-time computer vision project that detects hand gestures and facial proximity using **MediaPipe** and **OpenCV**. The system recognizes multiple gestures and triggers visual overlays based on live webcam input.

---

## Features

-  Real-time hand tracking using MediaPipe Hand Landmarker
-  Real-time face tracking using MediaPipe Face Landmarker
-  Gesture recognition (**So Far**):
    - Thumbs up 👍
    - Pointing up ☝️
    - Fist ✊
    - Hand near mouth as a fist 🫢
-  Gesture smoothing (prevents flickering using frame buffering)
-  Visual overlays triggered by detected gestures
-  Live webcam processing

---

##  How It Works

1. The webcam captures live video frame-by-frame.
2. MediaPipe processes each frame to extract:
   - Hand landmarks (21 points per hand)
   - Face landmarks (for mouth position)
3. Custom rule-based logic detects gestures based on landmark positions.
4. A frame-buffer system stabilizes gesture detection.
5. Matching gestures trigger overlay images displayed via OpenCV.

---

## Gesture Logic Overview

* 👍 Thumbs Up
    - All fingers folded except thumb
    - Thumb extended upward
* ☝️ Pointing Up
    - Index finger extended
    - Middle, ring and pinky fingers folded
* ✊ Fist
    - All fingers tucked
* 🫢 Hand Near Mouth
    - Distance between mouth center and hand center is below a threshold
    - All fingers tucked

---

## Technologies Used

- Python
- OpenCV
- MediaPipe Tasks API
- NumPy / Math (for distance calculations)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hand-tracking.git
cd hand-tracking
```

### 2. Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run it
```bash
python main.py
```

