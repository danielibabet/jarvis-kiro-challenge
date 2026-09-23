# Jarvis Vision — Best Practices

This skill provides guidelines and conventions for the Jarvis computer vision
module. Load it whenever working with `vision.py`, `control.py`, or any code
that touches the webcam, hand tracking, or desktop automation.

---

## 1. OpenCV — Mirror Correction

Always flip every captured frame horizontally immediately after `cap.read()`,
before any processing or rendering:

```python
ret, frame = cap.read()
frame = cv2.flip(frame, 1)  # 1 = horizontal flip (mirror effect)
```

Skipping this step causes the cursor to move in the opposite direction to the
user's hand, making the experience confusing and unusable.

---

## 2. MediaPipe — Correct API for Version ≥ 0.10

The legacy `mp.solutions.hands` sub-module was removed in MediaPipe 0.10+.
Always use the Tasks API with an explicit `.task` model file:

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

BaseOptions       = mp_tasks.BaseOptions
HandLandmarker    = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
```

The model file (`hand_landmarker.task`) must be downloaded once from:
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

---

## 3. PyAutoGUI — Smooth Movement with Bounds Validation

Never call `pyautogui.moveTo()` with raw, unvalidated coordinates.
Always pipe coordinates through `normalize_coordinates()` from `control.py`
first, then apply Exponential Moving Average (EMA) smoothing:

```python
# 1. Map normalised MediaPipe coords → absolute screen pixels (clamped)
mouse_x, mouse_y = normalize_coordinates(tip.x, tip.y, screen_width, screen_height)

# 2. EMA smoothing — avoids jitter and sudden jumps
SMOOTHING = 5  # higher = slower but smoother
curr_x = prev_x + (mouse_x - prev_x) / SMOOTHING
curr_y = prev_y + (mouse_y - prev_y) / SMOOTHING

# 3. Move — duration=0.0 because smoothing is handled by EMA, not PyAutoGUI
pyautogui.moveTo(curr_x, curr_y, duration=0.0)

prev_x, prev_y = curr_x, curr_y
```

`normalize_coordinates()` guarantees the result is always within
`[0, screen_width-1] × [0, screen_height-1]`, handling NaN, ±Inf, and
out-of-range values from MediaPipe edge cases.

---

## 4. Gesture Detection — Closed Fist

A finger is considered bent when its tip landmark Y-coordinate is greater than
its MCP (knuckle) Y-coordinate (Y grows downward in image space):

```python
# Tip/MCP pairs: index(8/5), middle(12/9), ring(16/13), pinky(20/17)
_FINGER_PAIRS = [(8, 5), (12, 9), (16, 13), (20, 17)]

def is_fist(hand_landmarks: list) -> bool:
    return all(
        hand_landmarks[tip].y > hand_landmarks[mcp].y
        for tip, mcp in _FINGER_PAIRS
    )
```

To avoid accidental triggers, require the fist to be held for at least
**1.5 seconds** and enforce a **10-second cooldown** between actions.

---

## 5. Testing — Property-Based Tests

Use **Hypothesis** for coordinate logic. The core invariant to always verify:

```python
from hypothesis import given, strategies as st
from control import normalize_coordinates

@given(
    x=st.floats(allow_nan=True, allow_infinity=True),
    y=st.floats(allow_nan=True, allow_infinity=True),
    w=st.integers(min_value=1, max_value=7680),
    h=st.integers(min_value=1, max_value=7680),
)
def test_always_in_bounds(x, y, w, h):
    px, py = normalize_coordinates(x, y, w, h)
    assert 0 <= px <= w - 1
    assert 0 <= py <= h - 1
```
