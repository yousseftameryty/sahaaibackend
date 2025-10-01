# preload_models.py
from transformers import pipeline
import subprocess
import time
import ollama  # pip install ollama (already in reqs)

# Preload HF model
pipeline("image-classification", model="nateraw/food")
print("HF model preloaded!")

# Start Ollama & pull
subprocess.Popen(["ollama", "serve"])  # Background
time.sleep(10)
ollama.pull("llama3.2:3b")
ollama.pull("llava:latest")
print("Ollama models pulled!")