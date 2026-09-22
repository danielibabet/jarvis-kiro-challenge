"""
control.py — Utilidades de control de escritorio para Jarvis.

Convierte coordenadas normalizadas de MediaPipe (0.0 – 1.0) a coordenadas
absolutas en píxeles, garantizando que el resultado esté siempre dentro de
los límites de la pantalla.
"""

import math


def normalize_coordinates(
    x_norm: float,
    y_norm: float,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """
    Mapea coordenadas normalizadas de MediaPipe a píxeles de pantalla.

    MediaPipe devuelve valores nominalmente en [0.0, 1.0], pero en la práctica
    puede emitir valores ligeramente fuera de ese rango (mano parcialmente
    fuera del encuadre) o incluso NaN/Inf en casos de pérdida de tracking.
    Esta función sanitiza cualquier entrada posible.

    Args:
        x_norm:        Coordenada X normalizada (nominalmente 0.0 – 1.0).
        y_norm:        Coordenada Y normalizada (nominalmente 0.0 – 1.0).
        screen_width:  Resolución horizontal de la pantalla en píxeles.
        screen_height: Resolución vertical de la pantalla en píxeles.

    Returns:
        Tupla (x_pixel, y_pixel) garantizada dentro de
        [0, screen_width - 1] × [0, screen_height - 1].

    Raises:
        ValueError: Si screen_width o screen_height son menores o iguales a 0.
    """
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError(
            f"Las dimensiones de pantalla deben ser positivas. "
            f"Recibido: screen_width={screen_width}, screen_height={screen_height}"
        )

    # --- Sanitizar NaN e Inf antes de cualquier operación aritmética ---
    # math.isfinite devuelve False para NaN, +Inf y -Inf.
    if not math.isfinite(x_norm):
        x_norm = 0.0 if math.isnan(x_norm) else (1.0 if x_norm > 0 else 0.0)
    if not math.isfinite(y_norm):
        y_norm = 0.0 if math.isnan(y_norm) else (1.0 if y_norm > 0 else 0.0)

    # --- Clampear al rango [0.0, 1.0] ---
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    # --- Escalar a píxeles ---
    # Se usa screen_width - 1 y screen_height - 1 para que el valor 1.0
    # produzca exactamente el último píxel válido (índice base-0).
    x_pixel = int(round(x_norm * (screen_width - 1)))
    y_pixel = int(round(y_norm * (screen_height - 1)))

    # --- Clampear píxeles (defensa en profundidad ante redondeos extremos) ---
    x_pixel = max(0, min(screen_width - 1, x_pixel))
    y_pixel = max(0, min(screen_height - 1, y_pixel))

    return (x_pixel, y_pixel)
