from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from datetime import datetime
import requests
import os
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app for chat
app = Flask(__name__)
CORS(app)

# Supabase configuration
SUPABASE_URL = "https://jtokmajtyrpmkhxnsmjc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp0b2ttYWp0eXJwbWtoeG5zbWpjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkxODYxNzcsImV4cCI6MjA3NDc2MjE3N30.5ukRWzCcGSzQOzyEXg1T_VPS3b-mZ8So0Dc22dY2Res"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class NutritionAIChat:
    def __init__(self):
        self.system_prompt = """You are SahaAI, a helpful and friendly nutrition assistant. You have access to the user's personal health profile and food scan history.

IMPORTANT GUIDELINES:
1. BE HELPFUL AND FRIENDLY - answer all questions about the user's own data
2. Use the user's profile data to provide personalized responses
3. When the user asks about THEIR OWN data (age, allergies, health info), share it freely
4. Only avoid giving medical treatment advice - but you CAN discuss nutrition, food safety, and general health
5. Be conversational and engaging

WHAT YOU CAN DO:
- Answer questions about the user's profile (age, weight, height, allergies, diabetes status)
- Provide nutrition advice based on their health conditions
- Discuss their food scan history
- Give general food safety and healthy eating tips
- Help with meal planning considering their allergies and diabetes
- Help with all food related questions no matter what they are (HAS TO APPLY TO OUR APP'S GOAL!)

BEHAVIOR:
- If user asks "What do you know about me?" - tell them their profile details
- If user asks "How old am I?" - tell them their age from the profile
- If user asks about allergies - list them and give avoidance tips
- Be warm, personal, and use their name
- If user asks about weather a food is good for them or not, be helpful and give them the information they need based on their profile, food scan history, and health conditions
- If user asks about their recent food scans, list them and give them the information they need based on their profile, food scan history, and health conditions
- If user asks about their health conditions, give them the information they need based on their profile, food scan history, and health conditions
"""

    def create_personalized_prompt(self, user_context, user_message, conversation_history="", user_memory=""):
        base_prompt = f"""{self.system_prompt}

USER CONTEXT:
- Name: {user_context.get('name', 'User')}
- Age: {user_context.get('age', 'Not specified')}
- Weight: {user_context.get('weight', 'Not specified')} kg
- Height: {user_context.get('height', 'Not specified')} cm
- Diabetes: {'Yes' if user_context.get('has_diabetes') else 'No'}
- Allergies: {', '.join(user_context.get('allergens', [])) or 'None'}
- Recent food scans: {', '.join(user_context.get('recent_scans', [])) or 'None'}

USER MEMORY (from previous conversations):
{user_memory if user_memory else 'No previous memories'}

{conversation_history}

Current message: {user_message}

Assistant:"""
        return base_prompt

    def clean_response(self, response):
        """Clean and format the AI response"""
        response = re.sub(r'Assistant:', '', response)
        response = re.sub(r'User:', '', response)
        response = response.strip()
        return response

# Initialize the chat AI
nutrition_chat = NutritionAIChat()

@app.route('/api/chat/health', methods=['GET'])
def chat_health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'SahaAI Chat Backend',
        'ollama_available': test_ollama_connection(),
        'database_connected': test_database_connection()
    })

@app.route('/api/chat/send-message', methods=['POST'])
def chat_send_message():
    try:
        data = request.json
        user_id = data.get('user_id')
        chat_id = data.get('chat_id')  # NEW: Support specific chat
        message = data.get('message')
        image_base64 = data.get('image_base64')
        conversation_history = data.get('conversation_history', '')

        if not user_id or (not message and not image_base64):
            return jsonify({'error': 'Missing user_id or message/image'}), 400

        logger.info(f"📱 Received chat message from user {user_id} in chat {chat_id}: {message or 'Image analysis'}")

        # Get user context from REAL database - MULTI-TABLE QUERY
        user_context = get_user_context(user_id)
        if not user_context:
            # Create a fallback user context if user not found
            user_context = {
                'name': 'User',
                'age': None,
                'weight': None,
                'height': None,
                'has_diabetes': False,
                'allergens': [],
                'recent_scans': [],
                'email': '',
                'profile_picture': None
            }

        # Get cross-chat memory for this user
        user_memory = get_user_memory(user_id, chat_id)
        
        # Create personalized prompt with memory
        user_message = message or "Please analyze this food image and provide nutritional information."
        prompt = nutrition_chat.create_personalized_prompt(
            user_context, 
            user_message, 
            conversation_history,
            user_memory
        )

        logger.info(f"🤖 Sending prompt to Ollama (length: {len(prompt)})")

        # Choose model based on whether there's an image
        if image_base64:
            logger.info("🖼️ Using vision model for image analysis")
            response = call_ollama_vision(prompt, image_base64)
        else:
            logger.info("📝 Using text model for chat")
            response = call_ollama_text(prompt)

        # Clean the response
        cleaned_response = nutrition_chat.clean_response(response)

        # Extract and save any new memories from the conversation
        save_memories_from_conversation(user_id, chat_id, user_message, cleaned_response)

        logger.info(f"✅ AI response generated (length: {len(cleaned_response)})")

        return jsonify({
            'response': cleaned_response,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        })

    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/history/<user_id>', methods=['GET'])
def get_chat_history(user_id):
    try:
        # Get chat history from REAL database
        history = get_user_chat_history(user_id)
        return jsonify({'history': history, 'success': True})
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/user/context/<user_id>', methods=['GET'])
def get_user_context_endpoint(user_id):
    try:
        context = get_user_context(user_id)
        if not context:
            return jsonify({'error': 'User not found', 'success': False}), 404
        return jsonify({'user_context': context, 'success': True})
    except Exception as e:
        logger.error(f"User context error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/create', methods=['POST'])
def create_new_chat():
    try:
        data = request.json
        user_id = data.get('user_id')
        initial_message = data.get('initial_message', 'Hello!')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Generate auto-name based on initial message
        auto_name = generate_chat_name(initial_message)
        
        # Create new chat WITH NAME
        chat_response = supabase.table('chats').insert({
            'user_id': user_id,
            'name': auto_name,  # ✅ STORE THE NAME IN DATABASE
            'last_message_at': datetime.utcnow().isoformat()
        }).execute()
        
        if not chat_response.data:
            return jsonify({'error': 'Failed to create chat'}), 500
        
        chat_id = chat_response.data[0]['id']
        
        logger.info(f"✅ Created new chat '{auto_name}' for user {user_id}")
        
        return jsonify({
            'chat_id': chat_id,
            'chat_name': auto_name,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"❌ Create chat error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/list/<user_id>', methods=['GET'])
def get_user_chats(user_id):
    try:
        # Get chats WITH PROPER FIELDS including name
        chats_response = supabase.table('chats')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('last_message_at', desc=True)\
            .execute()
        
        chats = []
        if chats_response.data:
            for chat in chats_response.data:
                # Get message count for this chat
                count_response = supabase.table('chat_messages')\
                    .select('id', count='exact')\
                    .eq('chat_id', chat['id'])\
                    .execute()
                
                message_count = count_response.count if hasattr(count_response, 'count') else 0
                
                chats.append({
                    'id': chat['id'],
                    'name': chat.get('name', f"Chat {chat['id']}"),  # ✅ Use stored name
                    'last_message_at': chat.get('last_message_at', chat['created_at']),
                    'message_count': message_count,
                    'is_active': chat.get('is_active', False)
                })
        
        return jsonify({'chats': chats, 'success': True})
        
    except Exception as e:
        logger.error(f"❌ Get chats error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/<chat_id>/rename', methods=['PUT'])
def rename_chat(chat_id):
    try:
        data = request.json
        new_name = data.get('name')
        user_id = data.get('user_id')
        
        if not new_name or not user_id:
            return jsonify({'error': 'Missing name or user_id'}), 400
        
        # ✅ ACTUALLY UPDATE THE NAME IN DATABASE
        update_response = supabase.table('chats')\
            .update({'name': new_name})\
            .eq('id', chat_id)\
            .eq('user_id', user_id)\
            .execute()
        
        if not update_response.data:
            return jsonify({'error': 'Failed to rename chat'}), 500
        
        logger.info(f"✅ Renamed chat {chat_id} to '{new_name}'")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"❌ Rename chat error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/<chat_id>/delete', methods=['DELETE'])
def delete_chat(chat_id):
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Verify user owns this chat
        chat_response = supabase.table('chats')\
            .select('user_id')\
            .eq('id', chat_id)\
            .execute()
        
        if not chat_response.data or chat_response.data[0]['user_id'] != user_id:
            return jsonify({'error': 'Chat not found or unauthorized'}), 404
        
        # Delete chat (cascade will handle messages)
        supabase.table('chats').delete().eq('id', chat_id).execute()
        
        logger.info(f"✅ Deleted chat {chat_id}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"❌ Delete chat error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/chat/<chat_id>/activate', methods=['POST'])
def activate_chat(chat_id):
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Verify user owns this chat
        chat_response = supabase.table('chats')\
            .select('user_id')\
            .eq('id', chat_id)\
            .execute()
        
        if not chat_response.data or chat_response.data[0]['user_id'] != user_id:
            return jsonify({'error': 'Chat not found or unauthorized'}), 404
        
        logger.info(f"✅ Activated chat {chat_id}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"❌ Activate chat error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# Helper functions
def test_ollama_connection():
    """Test if Ollama is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_database_connection():
    """Test if database connection works"""
    try:
        # Try to query a simple table
        result = supabase.table('users').select('count', count='exact').limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def get_user_context(user_id):
    """Get user profile and context from REAL database - MULTI-TABLE QUERIES"""
    try:
        logger.info(f"🔍 Fetching real user context for: {user_id}")
        
        # 1. Get user details from users table (main user data)
        user_response = supabase.table('users').select('*').eq('id', user_id).execute()
        
        if not user_response.data:
            logger.error(f"❌ User not found for: {user_id}")
            return None
        
        user_details = user_response.data[0]
        logger.info(f"✅ Found user: {user_details.get('email', 'Unknown')}")
        
        # 2. Get ALL allergies from user_allergens table
        allergens_response = supabase.table('user_allergens').select('allergen').eq('user_id', user_id).execute()
        individual_allergens = [item['allergen'] for item in allergens_response.data] if allergens_response.data else []
        
        logger.info(f"⚠️ User allergies: {individual_allergens}")
        
        # 3. Get recent food scans (last 5) from food_scans table
        scans_response = supabase.table('food_scans')\
            .select('food_name, created_at')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .limit(5)\
            .execute()
        
        recent_scans = [scan['food_name'] for scan in scans_response.data] if scans_response.data else []
        logger.info(f"📊 Found {len(recent_scans)} recent food scans: {recent_scans}")
        
        # 4. Build user context from ALL REAL data sources
        user_context = {
            'name': user_details.get('name', user_details.get('email', 'User').split('@')[0]),
            'age': user_details.get('age'),  # From users table
            'weight': user_details.get('weight'),  # From users table
            'height': user_details.get('height'),  # From users table
            'has_diabetes': user_details.get('has_diabetes', False),  # From users table
            'allergens': individual_allergens,  # From user_allergens table
            'recent_scans': recent_scans,
            'email': user_details.get('email', ''),
            'profile_picture': user_details.get('profile_picture')
        }
        
        logger.info(f"👤 COMPLETE user context: {user_context}")
        return user_context
        
    except Exception as e:
        logger.error(f"❌ Error getting user context: {e}")
        return None

def call_ollama_text(prompt, model="llama3.2:3b"):
    """Call Ollama for text generation"""
    try:
        logger.info(f"🔮 Calling Ollama text model: {model}")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', 'No response generated')
            logger.info(f"✅ Ollama text response received")
            return result
        else:
            error_msg = f"Ollama API error: {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Ollama text error: {e}")
        return "I'm having trouble processing your request right now. Please try again later."

def call_ollama_vision(prompt, image_base64, model="llava:latest"):
    """Call Ollama for vision analysis"""
    try:
        logger.info(f"🔮 Calling Ollama vision model: {model}")
        logger.info(f"📐 Image base64 length: {len(image_base64)}")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=1000
        )
        
        if response.status_code == 200:
            result = response.json().get('response', 'No response generated')
            logger.info(f"✅ Ollama vision response received")
            return result
        else:
            error_msg = f"Ollama vision API error: {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Ollama vision error: {e}")
        return "I'm having trouble analyzing the image right now. Please try again later."

def get_user_chat_history(user_id, chat_id=None, limit=20):
    """Get user chat history from REAL database"""
    try:
        logger.info(f"💬 Fetching real chat history for user: {user_id}")
        
        if chat_id:
            # Get messages for specific chat
            messages_response = supabase.table('chat_messages')\
                .select('*')\
                .eq('chat_id', chat_id)\
                .eq('user_id', user_id)\
                .order('created_at', ascending=True)\
                .limit(limit)\
                .execute()
        else:
            # Get user's latest chat
            chats_response = supabase.table('chats')\
                .select('id')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not chats_response.data:
                logger.info("📭 No chats found for user")
                return []
            
            chat_id = chats_response.data[0]['id']
            
            # Get messages for the latest chat
            messages_response = supabase.table('chat_messages')\
                .select('*')\
                .eq('chat_id', chat_id)\
                .eq('user_id', user_id)\
                .order('created_at', ascending=True)\
                .limit(limit)\
                .execute()
        
        history = []
        if messages_response.data:
            for msg in messages_response.data:
                history.append({
                    'id': msg['id'],
                    'content': msg['content'],
                    'isUser': msg['is_user'],
                    'timestamp': msg['created_at'],
                    'imageUri': msg.get('image_url')
                })
        
        logger.info(f"📨 Found {len(history)} chat messages")
        return history
        
    except Exception as e:
        logger.error(f"❌ Error getting chat history: {e}")
        return []

def generate_chat_name(initial_message):
    """Generate a simple 1-3 word name for a chat based on initial message"""
    try:
        # Simple keyword extraction for chat naming
        message_lower = initial_message.lower()
        
        # Food-related keywords
        food_keywords = ['food', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'recipe', 'cooking', 'nutrition', 'diet']
        health_keywords = ['health', 'fitness', 'weight', 'exercise', 'diabetes', 'allergy', 'allergies']
        question_keywords = ['help', 'advice', 'question', 'ask', 'tell', 'explain']
        
        # Find matching keywords
        found_keywords = []
        for keyword in food_keywords + health_keywords + question_keywords:
            if keyword in message_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            # Use the first 1-2 keywords found
            name_parts = found_keywords[:2]
            return ' '.join(name_parts).title()
        
        # Fallback: use first few words of the message
        words = initial_message.split()[:3]
        return ' '.join(words).title()
        
    except Exception as e:
        logger.error(f"❌ Error generating chat name: {e}")
        return "New Chat"

def get_user_memory(user_id, current_chat_id=None):
    """Get relevant memories for the user across all chats"""
    try:
        logger.info(f"🧠 Fetching user memory for: {user_id}")
        
        # Get all memories for this user
        memories_response = supabase.table('user_memory')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .limit(10)\
            .execute()
        
        if not memories_response.data:
            return ""
        
        # Format memories for the prompt
        memory_text = "Recent memories from our conversations:\n"
        for memory in memories_response.data:
            memory_text += f"- {memory['memory_type']}: {memory['content']}\n"
        
        logger.info(f"🧠 Found {len(memories_response.data)} memories")
        return memory_text
        
    except Exception as e:
        logger.error(f"❌ Error getting user memory: {e}")
        return ""

def save_memories_from_conversation(user_id, chat_id, user_message, ai_response):
    """Extract and save important information as memories"""
    try:
        logger.info(f"💾 Analyzing conversation for memories...")
        
        # Simple memory extraction logic
        memories_to_save = []
        
        # Check for preference mentions
        if any(word in user_message.lower() for word in ['like', 'love', 'prefer', 'favorite', 'hate', 'dislike']):
            memories_to_save.append({
                'memory_type': 'preference',
                'content': f"User mentioned: {user_message[:100]}...",
                'context': 'User preference expressed'
            })
        
        # Check for goal mentions
        if any(word in user_message.lower() for word in ['goal', 'want to', 'trying to', 'plan to', 'hope to']):
            memories_to_save.append({
                'memory_type': 'goal',
                'content': f"User goal: {user_message[:100]}...",
                'context': 'User goal mentioned'
            })
        
        # Check for new allergy information
        if any(word in user_message.lower() for word in ['allergic', 'allergy', 'intolerant', 'can\'t eat']):
            memories_to_save.append({
                'memory_type': 'allergy_update',
                'content': f"Allergy info: {user_message[:100]}...",
                'context': 'Allergy information shared'
            })
        
        # Save memories to database
        for memory_data in memories_to_save:
            supabase.table('user_memory').insert({
                'user_id': user_id,
                'chat_id': chat_id,
                'memory_type': memory_data['memory_type'],
                'content': memory_data['content'],
                'context': memory_data['context']
            }).execute()
        
        if memories_to_save:
            logger.info(f"💾 Saved {len(memories_to_save)} new memories")
        
    except Exception as e:
        logger.error(f"❌ Error saving memories: {e}")

if __name__ == '__main__':
    print("🚀 Starting SahaAI Chat Server...")
    print("📡 Server will run on http://192.168.1.9:5001")
    print("🔮 Make sure Ollama is running on http://localhost:11434")
    
    # Test connections
    if test_ollama_connection():
        print("✅ Ollama connection successful!")
    else:
        print("❌ Ollama connection failed! Make sure Ollama is running.")
    
    if test_database_connection():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed! Check Supabase credentials.")
    
    app.run(host='0.0.0.0', port=5001, debug=True)