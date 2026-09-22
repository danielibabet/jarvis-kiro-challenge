"""
test_control_pbt.py — Property-Based Tests para normalize_coordinates.

Usa Hypothesis para generar automáticamente miles de entradas aleatorias
(incluyendo valores negativos, muy grandes, NaN e Inf) y verifica que la
propiedad fundamental se cumpla siempre:

    0 <= x_pixel <= screen_width - 1
    0 <= y_pixel <= screen_height - 1

Ejecutar:
    python -m pytest test_control_pbt.py -v
"""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from control import normalize_coordinates

# ---------------------------------------------------------------------------
# Estrategias de generación de datos
# ---------------------------------------------------------------------------

# Flotantes sin restricciones: incluye negativos, muy grandes, NaN e Inf.
any_float = st.floats(allow_nan=True, allow_infinity=True)

# Flotantes finitos y en rango nominal de MediaPipe [0.0, 1.0].
nominal_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Flotantes finitos pero fuera del rango nominal.
out_of_range_float = st.floats(allow_nan=False, allow_infinity=False).filter(
    lambda v: v < 0.0 or v > 1.0
)

# Resoluciones de pantalla realistas (1 px mínimo, hasta 8K).
screen_dim = st.integers(min_value=1, max_value=7680)


# ---------------------------------------------------------------------------
# Propiedad 1 — Invariante principal (cualquier flotante posible)
# ---------------------------------------------------------------------------
@given(
    x_norm=any_float,
    y_norm=any_float,
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=10_000)
def test_resultado_siempre_dentro_de_limites(
    x_norm: float, y_norm: float, screen_width: int, screen_height: int
) -> None:
    """
    Para CUALQUIER entrada de punto flotante, el resultado siempre debe
    estar dentro de [0, screen_width - 1] × [0, screen_height - 1].
    """
    x_pixel, y_pixel = normalize_coordinates(x_norm, y_norm, screen_width, screen_height)

    assert 0 <= x_pixel <= screen_width - 1, (
        f"x_pixel={x_pixel} fuera de [0, {screen_width - 1}] "
        f"para x_norm={x_norm!r}, screen_width={screen_width}"
    )
    assert 0 <= y_pixel <= screen_height - 1, (
        f"y_pixel={y_pixel} fuera de [0, {screen_height - 1}] "
        f"para y_norm={y_norm!r}, screen_height={screen_height}"
    )


# ---------------------------------------------------------------------------
# Propiedad 2 — Monotonicidad (entradas en rango nominal [0.0, 1.0])
# ---------------------------------------------------------------------------
@given(
    x1=nominal_float,
    x2=nominal_float,
    y=nominal_float,
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=5_000)
def test_monotonicidad_x(
    x1: float, x2: float, y: float, screen_width: int, screen_height: int
) -> None:
    """
    Si x1 <= x2 (ambos en [0,1]), entonces x_pixel1 <= x_pixel2.
    El mapeo debe ser no decreciente.
    """
    px1, _ = normalize_coordinates(x1, y, screen_width, screen_height)
    px2, _ = normalize_coordinates(x2, y, screen_width, screen_height)

    if x1 <= x2:
        assert px1 <= px2, (
            f"Monotonicidad rota: x1={x1} → px1={px1}, x2={x2} → px2={px2}"
        )


@given(
    x=nominal_float,
    y1=nominal_float,
    y2=nominal_float,
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=5_000)
def test_monotonicidad_y(
    x: float, y1: float, y2: float, screen_width: int, screen_height: int
) -> None:
    """
    Si y1 <= y2 (ambos en [0,1]), entonces y_pixel1 <= y_pixel2.
    """
    _, py1 = normalize_coordinates(x, y1, screen_width, screen_height)
    _, py2 = normalize_coordinates(x, y2, screen_width, screen_height)

    if y1 <= y2:
        assert py1 <= py2, (
            f"Monotonicidad rota: y1={y1} → py1={py1}, y2={y2} → py2={py2}"
        )


# ---------------------------------------------------------------------------
# Propiedad 3 — Clamp simétrico (fuera de rango produce el mismo extremo)
# ---------------------------------------------------------------------------
@given(
    x_neg=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
    y_neg=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=3_000)
def test_valores_negativos_producen_origen(
    x_neg: float, y_neg: float, screen_width: int, screen_height: int
) -> None:
    """
    Cualquier valor negativo debe mapearse a (0, 0).
    """
    x_pixel, y_pixel = normalize_coordinates(x_neg, y_neg, screen_width, screen_height)
    assert x_pixel == 0, f"x_norm={x_neg!r} debería producir x_pixel=0, obtuvo {x_pixel}"
    assert y_pixel == 0, f"y_norm={y_neg!r} debería producir y_pixel=0, obtuvo {y_pixel}"


@given(
    x_big=st.floats(min_value=1.0 + 1e-9, allow_nan=False, allow_infinity=False),
    y_big=st.floats(min_value=1.0 + 1e-9, allow_nan=False, allow_infinity=False),
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=3_000)
def test_valores_mayores_que_uno_producen_maximo(
    x_big: float, y_big: float, screen_width: int, screen_height: int
) -> None:
    """
    Cualquier valor > 1.0 debe mapearse al último píxel válido.
    """
    x_pixel, y_pixel = normalize_coordinates(x_big, y_big, screen_width, screen_height)
    assert x_pixel == screen_width - 1, (
        f"x_norm={x_big!r} debería producir x_pixel={screen_width - 1}, obtuvo {x_pixel}"
    )
    assert y_pixel == screen_height - 1, (
        f"y_norm={y_big!r} debería producir y_pixel={screen_height - 1}, obtuvo {y_pixel}"
    )


# ---------------------------------------------------------------------------
# Propiedad 4 — NaN siempre produce el origen (comportamiento seguro)
# ---------------------------------------------------------------------------
@given(
    screen_width=screen_dim,
    screen_height=screen_dim,
)
@settings(max_examples=1_000)
def test_nan_produce_origen(screen_width: int, screen_height: int) -> None:
    """
    NaN en cualquier coordenada debe producir 0 para esa dimensión.
    """
    nan = float("nan")

    x_pixel, y_pixel = normalize_coordinates(nan, nan, screen_width, screen_height)
    assert x_pixel == 0, f"NaN en X debería producir 0, obtuvo {x_pixel}"
    assert y_pixel == 0, f"NaN en Y debería producir 0, obtuvo {y_pixel}"

    # NaN solo en X, Y válida
    x_pixel, y_pixel = normalize_coordinates(nan, 0.5, screen_width, screen_height)
    assert x_pixel == 0, f"NaN en X debería producir 0, obtuvo {x_pixel}"
    assert 0 <= y_pixel <= screen_height - 1

    # NaN solo en Y, X válida
    x_pixel, y_pixel = normalize_coordinates(0.5, nan, screen_width, screen_height)
    assert 0 <= x_pixel <= screen_width - 1
    assert y_pixel == 0, f"NaN en Y debería producir 0, obtuvo {y_pixel}"


# ---------------------------------------------------------------------------
# Propiedad 5 — ValueError para dimensiones de pantalla inválidas
# ---------------------------------------------------------------------------
@given(
    x_norm=any_float,
    y_norm=any_float,
    bad_dim=st.integers(max_value=0),
)
@settings(max_examples=2_000)
def test_dimensiones_invalidas_lanzan_valueerror(
    x_norm: float, y_norm: float, bad_dim: int
) -> None:
    """
    screen_width o screen_height <= 0 debe lanzar ValueError, sin importar
    las coordenadas de entrada.
    """
    with pytest.raises(ValueError):
        normalize_coordinates(x_norm, y_norm, bad_dim, 1080)

    with pytest.raises(ValueError):
        normalize_coordinates(x_norm, y_norm, 1920, bad_dim)
