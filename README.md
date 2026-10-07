# Jarvis - Local AI Desktop Assistant with Computer Vision

<p align="center">
  <img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/Ollama_Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
</p>

<p align="center">
  </a>
</p>

Developed for the **Kiro University Challenge**. Jarvis is an autonomous local AI assistant that combines open-source Large Language Models (LLMs) with computer vision gesture tracking to control your desktop interface without a physical mouse or keyboard.

---

## Core Architecture & Stack

- **Local LLM Intelligence:** Powered by Ollama running open-source instruction-tuned models (e.g., Qwen / Gemma) locally for private, low-latency reasoning.
- **Computer Vision & Hand Tracking:** Real-time gesture and landmark recognition using **OpenCV** and **MediaPipe**.
- **Desktop Control & Automation:** **PyAutoGUI** coordinate mapping for virtual cursor motion, clicks, drag-and-drop, and shortcuts.
- **Testing & Quality Assurance:** Property-Based Testing for coordinate boundary validation and automation hooks.

---

## Hardware & System Requirements

- Webcam and microphone.
- Multi-core CPU / GPU capable of local LLM inference via Ollama.
- Python 3.10+.

---

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/danielibabet/jarvis-kiro-challenge.git
cd jarvis-kiro-challenge

# Install Python dependencies
pip install opencv-python mediapipe pyautogui requests

# Ensure Ollama is running locally
ollama run qwen2.5:latest

# Run Jarvis
python vision.py
```

---

## Support & Author

- **Daniel Ibáñez** - [@danielibabet](https://github.com/danielibabet)
