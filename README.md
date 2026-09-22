# Jarvis - Asistente de IA Local y Control por Visión Artificial

Proyecto desarrollado para el **Kiro University Challenge (Septiembre 2026)**. 
Jarvis es un asistente local que combina modelos de lenguaje de código abierto con control de escritorio mediante visión artificial, permitiendo interactuar con el sistema operativo sin ratón ni teclado convencionales.

## Arquitectura y Stack Tecnológico
* **Cerebro (LLM Local):** Ollama ejecutando modelos de código abierto (Qwen / Gemma).
* **Visión y Reconocimiento:** OpenCV y MediaPipe para el rastreo de gestos y manos a través de la webcam.
* **Control de Escritorio:** PyAutoGUI para la ejecución de macros, movimiento del cursor y control de la interfaz.
* **Desarrollo y Testing:** Kiro IDE (implementando Hooks de automatización y Property-Based Testing para la validación de coordenadas).

## Requisitos de Hardware
* PC con cámara web y micrófono.
* Recursos suficientes para la ejecución local de modelos LLM.

## Autor
Daniel Ibáñez Betés