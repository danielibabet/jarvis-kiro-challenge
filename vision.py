"""
vision.py — Detección de manos con MediaPipe Tasks API (≥ 0.10 / 1.x) y OpenCV.

La API legacy `mp.solutions` fue eliminada en MediaPipe 0.10+.
Ahora se usa `mediapipe.tasks.python.vision.HandLandmarker` con un fichero
de modelo .task que se descarga automáticamente si no está presente.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks import python as mp_tasks

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Índices de landmarks (igual que la API legacy)
INDEX_FINGER_TIP = 8

# Conexiones entre landmarks para dibujar el esqueleto de la mano manualmente
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # corazón
    (9, 13), (13, 14), (14, 15), (15, 16), # anular
    (13, 17), (17, 18), (18, 19), (19, 20),# meñique
    (0, 17),                               # palma
]


# ---------------------------------------------------------------------------
# Descarga del modelo (solo la primera vez)
# ---------------------------------------------------------------------------
def _ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"Descargando modelo en {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado correctamente.")
    return str(MODEL_PATH)


# ---------------------------------------------------------------------------
# Función pública requerida
# ---------------------------------------------------------------------------
def get_index_finger_coordinates(
    hand_landmarks: list, frame_width: int, frame_height: int
) -> tuple[int, int]:
    """
    Devuelve las coordenadas en píxeles de la punta del dedo índice.

    Args:
        hand_landmarks: Lista de NormalizedLandmark de una mano (21 puntos).
        frame_width:    Ancho del frame capturado (en píxeles).
        frame_height:   Alto del frame capturado (en píxeles).

    Returns:
        Tupla (x, y) con la posición absoluta de la punta del índice.
    """
    tip = hand_landmarks[INDEX_FINGER_TIP]
    x = int(tip.x * frame_width)
    y = int(tip.y * frame_height)
    return (x, y)


# ---------------------------------------------------------------------------
# Helpers de dibujo
# ---------------------------------------------------------------------------
def _draw_hand(frame: "cv2.Mat", landmarks: list, width: int, height: int) -> None:
    """Dibuja los 21 landmarks y las conexiones sobre el frame."""
    # Puntos
    points: dict[int, tuple[int, int]] = {}
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * width), int(lm.y * height)
        points[idx] = (cx, cy)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    # Conexiones
    for start, end in HAND_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (0, 200, 255), 2)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------
def main() -> None:
    model_path = _ensure_model()

    BaseOptions = mp_tasks.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara web.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame. Saliendo…")
                break

            frame_height, frame_width = frame.shape[:2]
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Convierte BGR → RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    _draw_hand(frame, hand_landmarks, frame_width, frame_height)

                    x, y = get_index_finger_coordinates(
                        hand_landmarks, frame_width, frame_height
                    )
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"Indice: ({x}, {y})",
                        (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Jarvis - Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
