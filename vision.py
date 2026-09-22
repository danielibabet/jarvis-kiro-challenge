"""
vision.py — Detección de manos con MediaPipe Tasks API (≥ 0.10 / 1.x) y OpenCV.

La API legacy `mp.solutions` fue eliminada en MediaPipe 0.10+.
Ahora se usa `mediapipe.tasks.python.vision.HandLandmarker` con un fichero
de modelo .task que se descarga automáticamente si no está presente.

El dedo índice de la primera mano detectada controla el cursor del ratón.
Las coordenadas normalizadas de MediaPipe se convierten a píxeles de pantalla
mediante normalize_coordinates (control.py) y se aplican con pyautogui.moveTo.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from brain import ask_ollama
from control import normalize_coordinates

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

# Factor de suavizado del movimiento del ratón.
# Cuanto mayor sea el valor, más lento y suave es el movimiento.
SMOOTHING = 5

# Tiempo (segundos) que el puño debe mantenerse cerrado para activar Jarvis
FIST_HOLD = 1.5

# Cooldown (segundos) entre capturas consecutivas para no saturar Ollama
JARVIS_COOLDOWN = 10

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
# Detección de gesto: puño cerrado
# ---------------------------------------------------------------------------
# Pares (tip, mcp) para índice, medio, anular y meñique.
# Un dedo está doblado cuando su punta (tip) queda por debajo de su nudillo
# proximal (MCP) en el eje Y de la imagen (Y crece hacia abajo).
_FINGER_PAIRS = [(8, 5), (12, 9), (16, 13), (20, 17)]


def is_fist(hand_landmarks: list) -> bool:
    """
    Devuelve True si los cuatro dedos (índice, medio, anular, meñique)
    están doblados hacia la palma, formando un puño cerrado.

    Args:
        hand_landmarks: Lista de 21 NormalizedLandmark de una mano.

    Returns:
        True si todos los dedos están doblados, False en caso contrario.
    """
    return all(
        hand_landmarks[tip].y > hand_landmarks[mcp].y
        for tip, mcp in _FINGER_PAIRS
    )


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

    # Resolución real de la pantalla (ej. 2560×1440)
    screen_width, screen_height = pyautogui.size()

    # Desactiva la protección fail-safe de pyautogui para que el movimiento
    # del ratón no se interrumpa al llegar a las esquinas.
    # Cámbialo a True si quieres poder abortar moviendo el ratón a (0, 0).
    pyautogui.FAILSAFE = False

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

    # Posición suavizada del ratón; se inicializa en el centro de la pantalla
    prev_x = screen_width // 2
    prev_y = screen_height // 2

    # Estado para la detección de puño sostenido
    fist_start: float | None = None   # timestamp en que comenzó el puño
    last_capture: float = 0.0         # timestamp de la última captura enviada a Ollama

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame. Saliendo…")
                break

            # Invertir horizontalmente para efecto espejo natural
            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Convierte BGR → RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                # Solo la primera mano controla el ratón para evitar conflictos
                primary_hand = result.hand_landmarks[0]

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

                # Obtener coordenadas normalizadas del índice de la mano primaria
                # y mapearlas a la resolución de la pantalla
                tip = primary_hand[INDEX_FINGER_TIP]
                mouse_x, mouse_y = normalize_coordinates(
                    tip.x, tip.y, screen_width, screen_height
                )

                # Suavizado exponencial: interpola entre la posición anterior
                # y la nueva usando el factor SMOOTHING (EMA)
                curr_x = prev_x + (mouse_x - prev_x) / SMOOTHING
                curr_y = prev_y + (mouse_y - prev_y) / SMOOTHING

                pyautogui.moveTo(curr_x, curr_y, duration=0.0)

                prev_x = curr_x
                prev_y = curr_y

                # --- Detección de puño y trigger de Jarvis ---
                now = time.time()
                if is_fist(primary_hand):
                    if fist_start is None:
                        fist_start = now  # inicio del gesto

                    elapsed = now - fist_start
                    cooldown_remaining = JARVIS_COOLDOWN - (now - last_capture)

                    # Barra de progreso visual (rojo) en la parte inferior del frame
                    progress = min(elapsed / FIST_HOLD, 1.0)
                    bar_w = int(200 * progress)
                    cv2.rectangle(frame, (10, frame_height - 30), (210, frame_height - 10), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, frame_height - 30), (10 + bar_w, frame_height - 10), (0, 0, 255), -1)
                    cv2.putText(frame, "Puno", (215, frame_height - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    if elapsed >= FIST_HOLD and cooldown_remaining <= 0:
                        print("[Jarvis] Puño detectado — capturando pantalla…")
                        screenshot_path = "pantalla.png"
                        pyautogui.screenshot(screenshot_path)

                        print("[Jarvis] Consultando a Ollama…")
                        respuesta = ask_ollama(
                            "Eres Jarvis. Describe de forma muy breve y directa lo que ves en esta interfaz",
                            image_path=screenshot_path,
                        )
                        print(f"[Jarvis] {respuesta}")

                        last_capture = now   # reinicia el cooldown
                        fist_start = None    # requiere soltar y volver a cerrar el puño
                else:
                    fist_start = None  # se soltó el puño antes de tiempo

            else:
                # Sin mano detectada: resetear temporizador de puño
                fist_start = None

            cv2.imshow("Jarvis - Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
