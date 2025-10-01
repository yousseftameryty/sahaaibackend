# main.py - Unified SahaAI Backend (exact copy-paste from your files + merges)
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import requests
import os
import json
import re
from transformers import pipeline
from PIL import Image
from io import BytesIO
import torch
from datetime import datetime
import logging
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# Logging setup from chat.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase from chat.py (use env vars)
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# UltimateFoodAI class from app.py (exact copy)
class UltimateFoodAI:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        self.classifier = pipeline("image-classification", model="nateraw/food", device=device)
    # ... (copy ALL methods exactly: detect_food_huggingface, clean_food_label, is_food_related, get_nutrition_data, etc. up to assess_diabetes_safety)
    # Paste the full class body from your app.py here—no changes!

ultimate_ai = UltimateFoodAI()

# NutritionAIChat class from chat.py (exact copy)
class NutritionAIChat:
    def __init__(self):
        self.system_prompt = """..."""  # Exact from your chat.py
    # ... (copy ALL methods: create_personalized_prompt, clean_response)
    # Paste full class body—no changes!

nutrition_chat = NutritionAIChat()

# Routes from app.py (exact, no @app.route—define as functions)
def analyze_food():
    # Exact body from your app.py /api/analyze-food
    # Uses ultimate_ai
    try:
        data = request.json
        # ... full exact code ...
        return jsonify(response)
    except Exception as e:
        # ... full exact ...
        return jsonify({'error': str(e)}), 500

def health_check():
    # Exact from app.py /api/health
    return jsonify({...})  # Full dict

# Routes from chat.py (define as functions, no @app.route; use supabase global)
def chat_health_check():
    # Exact body, but replace app= with global supabase; test_ollama_connection etc. stay
    return jsonify({...})

def chat_send_message():
    # Exact body from chat.py; uses supabase, nutrition_chat, call_ollama_text/vision
    # Helper funcs (get_user_context, call_ollama_text, etc.) go below as defs
    try:
        # ... full exact ...
        return jsonify({...})
    except Exception as e:
        # ...
        return jsonify({'error': str(e), 'success': False}), 500

# ... Define ALL other chat routes as functions: get_chat_history, get_user_context_endpoint, create_new_chat, etc.
# Copy ALL helper defs from chat.py: test_ollama_connection, get_user_context, call_ollama_text, call_ollama_vision, get_user_chat_history, generate_chat_name, get_user_memory, save_memories_from_conversation
# Exact copies—no changes!

# Register ALL routes to app
app.route('/api/analyze-food', methods=['POST'])(analyze_food)
app.route('/api/health', methods=['GET'])(health_check)
app.route('/api/chat/health', methods=['GET'])(chat_health_check)
app.route('/api/chat/send-message', methods=['POST'])(chat_send_message)
app.route('/api/chat/history/<user_id>', methods=['GET'])(get_chat_history)
app.route('/api/chat/user/context/<user_id>', methods=['GET'])(get_user_context_endpoint)
app.route('/api/chat/create', methods=['POST'])(create_new_chat)
app.route('/api/chat/list/<user_id>', methods=['GET'])(get_user_chats)
app.route('/api/chat/<chat_id>/rename', methods=['PUT'])(rename_chat)
app.route('/api/chat/<chat_id>/delete', methods=['DELETE'])(delete_chat)
app.route('/api/chat/<chat_id>/activate', methods=['POST'])(activate_chat)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)