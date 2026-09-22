"""
brain.py — Interfaz con el LLM local mediante Ollama.

Expone ask_ollama(), que envía un prompt (opcionalmente acompañado de una
imagen) al modelo qwen2.5:7b corriendo en Ollama y devuelve el texto limpio
de la respuesta.

Requisito previo:
    ollama serve          # servidor escuchando en localhost:11434
    ollama pull qwen2.5:7b
"""

import base64
import json
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------
def ask_ollama(prompt: str, image_path: str | None = None) -> str:
    """
    Envía un prompt al modelo local qwen2.5:7b a través de Ollama.

    Si se proporciona image_path, la imagen se codifica en base64 y se adjunta
    en el payload para que el modelo pueda interpretarla visualmente.

    Args:
        prompt:     Texto de la instrucción o pregunta al modelo.
        image_path: Ruta opcional a un archivo de imagen (PNG, JPG, etc.).

    Returns:
        Texto limpio de la respuesta generada por el modelo.

    Raises:
        FileNotFoundError: Si image_path no apunta a un archivo existente.
        requests.HTTPError: Si Ollama devuelve un código de error HTTP.
        requests.ConnectionError: Si el servidor Ollama no está disponible.
    """
    payload: dict = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,  # respuesta completa en un único JSON
    }

    # --- Adjuntar imagen si se proporciona ---
    if image_path is not None:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        # Ollama espera una lista de strings base64 bajo la clave "images"
        payload["images"] = [encoded]

    # --- Petición POST ---
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    # --- Extraer texto limpio ---
    # Con stream=False Ollama devuelve un único objeto JSON con la clave "response"
    data = response.json()
    return data.get("response", "").strip()


# ---------------------------------------------------------------------------
# Ejecución directa para pruebas rápidas
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else None
    question = sys.argv[2] if len(sys.argv) > 2 else "Hola, ¿qué puedes hacer?"

    print(f"Modelo : {MODEL}")
    print(f"Prompt : {question}")
    print(f"Imagen : {img or 'ninguna'}")
    print("-" * 40)

    try:
        answer = ask_ollama(question, image_path=img)
        print(answer)
    except requests.ConnectionError:
        print("ERROR: No se puede conectar con Ollama. ¿Está ejecutándose 'ollama serve'?")
    except requests.HTTPError as e:
        print(f"ERROR HTTP: {e}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
