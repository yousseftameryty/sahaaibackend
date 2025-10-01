# app.py - Unified SahaAI Backend (full merge of original app.py + chat.py)
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

# UltimateFoodAI class from original app.py (full exact)
class UltimateFoodAI:
    def __init__(self):
        # Detect device: Use GPU if available (for GTX 1650), else CPU
        device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        # Load the Hugging Face pipeline once for efficiency
        self.classifier = pipeline("image-classification", model="nateraw/food", device=device)

    def detect_food_microsoft(self, image_data):
        """Use Microsoft Computer Vision - but skipped in fixed version as it requires key."""
        return self.detect_food_huggingface(image_data)  # Fallback to local HF

    def clean_food_label(self, label):
        """Clean food labels from Hugging Face model"""
        # Food-101 labels are like: "apple_pie", "baby_back_ribs", etc.
        cleaned = label.replace('_', ' ').title()
        
        # Fix common label issues
        corrections = {
            'Baby Back Ribs': 'Ribs',
            'Prime Rib': 'Beef Rib',
            'Bread Pudding': 'Bread Pudding',
            'Cheese Cake': 'Cheesecake',
            'French Fries': 'French Fries',
            'Grilled Cheese Sandwich': 'Grilled Cheese',
            'Hot Dog': 'Hot Dog',
            'Ice Cream': 'Ice Cream',
            'Spring Rolls': 'Spring Rolls',
            'Tuna Tartare': 'Tuna'
        }
        
        return corrections.get(cleaned, cleaned)
    
    def detect_food_huggingface(self, image_data):
        """Use local Hugging Face model for food detection - free, no limits."""
        try:
            # Clean image data
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Run classification
            results = self.classifier(image)
            
            food_items = []
            for res in results:
                label = res['label']
                if self.is_food_related(label.lower()):
                    food_items.append({
                        'name': self.clean_food_label(label),
                        'confidence': res['score'],
                        'type': 'huggingface'
                    })
            
            return food_items
            
        except Exception as e:
            print(f"Hugging Face error: {e}")
            return []  # Return empty on error
    
    def is_food_related(self, label):
        """Check if a label is food-related - EXPANDED & MORE LENIENT"""
        food_keywords = [
            # Fruits
            'apple', 'banana', 'orange', 'fruit', 'grape', 'strawberry', 'blueberry', 
            'raspberry', 'watermelon', 'pineapple', 'mango', 'peach', 'pear', 'plum',
            'cherry', 'kiwi', 'lemon', 'lime', 'coconut', 'avocado',
            
            # Vegetables
            'vegetable', 'carrot', 'broccoli', 'tomato', 'potato', 'onion', 'garlic',
            'lettuce', 'spinach', 'kale', 'cabbage', 'cauliflower', 'celery', 'cucumber',
            'pepper', 'bell pepper', 'chili', 'mushroom', 'corn', 'pea',
            
            # Proteins
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp', 'egg', 'eggs',
            'meat', 'steak', 'bacon', 'sausage', 'turkey', 'duck', 'lamb',
            
            # Dairy
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'ice cream',
            
            # Grains & Carbs
            'bread', 'rice', 'pasta', 'noodle', 'potato', 'cereal', 'oatmeal', 'flour',
            'wheat', 'grain', 'corn', 'quinoa',
            
            # Common foods & meals
            'food', 'meal', 'dish', 'cuisine', 'pizza', 'burger', 'sandwich', 'soup',
            'salad', 'curry', 'stew', 'sauce', 'dessert', 'cake', 'cookie', 'chocolate',
            'snack', 'breakfast', 'lunch', 'dinner',
            
            # Drinks
            'drink', 'beverage', 'coffee', 'tea', 'juice', 'smoothie', 'water',
            
            # Cooking terms
            'cooked', 'fried', 'grilled', 'baked', 'roasted', 'boiled', 'raw'
        ]
        
        label_lower = label.lower()
        
        # Direct match
        if any(food == label_lower for food in food_keywords):
            return True
            
        # Partial match (more lenient)
        if any(food in label_lower for food in food_keywords):
            return True
            
        # Common food patterns
        food_patterns = ['food', 'fruit', 'vegetable', 'meal', 'dish', 'snack']
        if any(pattern in label_lower for pattern in food_patterns):
            return True
            
        return False
    
    def get_nutrition_data(self, food_name):
        """Get nutrition data with better matching"""
        print(f"🍎 Getting nutrition for: {food_name}")
        
        # Clean food name
        food_clean = self.clean_food_name(food_name)
        
        # Try USDA first
        nutrition_data = self.get_nutrition_usda(food_clean)
        if nutrition_data and nutrition_data.get('calories', 0) > 0:
            return nutrition_data
        
        # Try Open Food Facts
        nutrition_data = self.get_nutrition_open_food_facts(food_clean)
        if nutrition_data and nutrition_data.get('calories', 0) > 0:
            return nutrition_data
        
        # Use comprehensive database with better matching
        nutrition_data = self.get_comprehensive_nutrition(food_clean)
        return nutrition_data
    
    def clean_food_name(self, food_name):
        """Clean food name for better matching"""
        # Remove common prefixes/suffixes
        clean_name = food_name.lower()
        removals = ['fresh', 'raw', 'cooked', 'ripe', 'organic', 'whole', 'slice', 'slices', 'piece', 'pieces']
        
        for removal in removals:
            clean_name = clean_name.replace(removal, '').strip()
        
        # Remove extra spaces
        clean_name = ' '.join(clean_name.split())
        
        return clean_name if clean_name else food_name.lower()
    
    def get_nutrition_usda(self, food_name):
        """USDA FoodData Central - FIXED"""
        try:
            # Using public demo key
            params = {
                'api_key': 'DEMO_KEY',
                'query': food_name,
                'pageSize': 3,
                'dataType': ['Foundation', 'SR Legacy', 'Survey (FNDDS)']
            }
            
            response = requests.get(
                "https://api.nal.usda.gov/fdc/v1/foods/search", 
                params=params, 
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('foods') and len(data['foods']) > 0:
                    best_match = data['foods'][0]
                    
                    nutrients = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugars': 0}
                    
                    for nutrient in best_match.get('foodNutrients', []):
                        name = nutrient.get('nutrientName', '').lower()
                        value = nutrient.get('value', 0)
                        unit = nutrient.get('unitName', '').lower()
                        
                        if 'energy' in name or 'calorie' in name:
                            if unit == 'kj':
                                value = value / 4.184
                            nutrients['calories'] = value
                        elif 'protein' in name:
                            nutrients['protein'] = value
                        elif 'carbohydrate' in name and 'total' in name:
                            nutrients['carbs'] = value
                        elif 'total fat' in name or 'fat' in name and 'total' in name:
                            nutrients['fat'] = value
                        elif 'fiber' in name and 'dietary' in name:
                            nutrients['fiber'] = value
                        elif 'sugar' in name and 'total' in name:
                            nutrients['sugars'] = value
                    
                    if nutrients['calories'] > 0:
                        print(f"✅ USDA data found: {nutrients['calories']} calories")
                        return nutrients
            
            return None
            
        except Exception as e:
            print(f"USDA API error: {e}")
            return None
    
    def get_nutrition_open_food_facts(self, food_name):
        """Open Food Facts - FIXED"""
        try:
            params = {
                'search_terms': food_name,
                'json': 1,
                'page_size': 3,
                'fields': 'product_name,nutriments,categories'
            }
            
            response = requests.get(
                "https://world.openfoodfacts.org/cgi/search.pl", 
                params=params, 
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('products') and len(data['products']) > 0:
                    product = data['products'][0]
                    nutriments = product.get('nutriments', {})
                    
                    nutrients = {
                        'calories': nutriments.get('energy-kcal_100g', nutriments.get('energy_100g', 0) / 4.184),
                        'protein': nutriments.get('proteins_100g', 0),
                        'carbs': nutriments.get('carbohydrates_100g', 0),
                        'fat': nutriments.get('fat_100g', 0),
                        'fiber': nutriments.get('fiber_100g', 0),
                        'sugars': nutriments.get('sugars_100g', 0)
                    }
                    
                    if nutrients['calories'] > 0:
                        print(f"✅ Open Food Facts data found: {nutrients['calories']} calories")
                        return nutrients
            
            return None
            
        except Exception as e:
            print(f"Open Food Facts error: {e}")
            return None
    
    def get_comprehensive_nutrition(self, food_name):
        """Comprehensive nutrition database - EXPANDED & SMARTER MATCHING"""
        comprehensive_db = {
            'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4, 'sugars': 10},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6, 'sugars': 12},
            'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4, 'sugars': 9},
            'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 8, 'fat': 0.3, 'fiber': 2, 'sugars': 4.9},
            'grape': {'calories': 69, 'protein': 0.7, 'carbs': 18, 'fat': 0.2, 'fiber': 0.9, 'sugars': 16},
            'watermelon': {'calories': 30, 'protein': 0.6, 'carbs': 8, 'fat': 0.2, 'fiber': 0.4, 'sugars': 6},
            'pineapple': {'calories': 50, 'protein': 0.5, 'carbs': 13, 'fat': 0.1, 'fiber': 1.4, 'sugars': 10},
            'mango': {'calories': 60, 'protein': 0.8, 'carbs': 15, 'fat': 0.4, 'fiber': 1.6, 'sugars': 14},
            'pear': {'calories': 57, 'protein': 0.4, 'carbs': 15, 'fat': 0.1, 'fiber': 3.1, 'sugars': 10},
            'peach': {'calories': 39, 'protein': 0.9, 'carbs': 10, 'fat': 0.3, 'fiber': 1.5, 'sugars': 8},
            'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8, 'sugars': 4.7},
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6, 'sugars': 1.7},
            'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2, 'sugars': 2.6},
            'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'fiber': 2.2, 'sugars': 0.8},
            'onion': {'calories': 40, 'protein': 1.1, 'carbs': 9, 'fat': 0.1, 'fiber': 1.7, 'sugars': 4.2},
            'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'fiber': 2.2, 'sugars': 0.4},
            'lettuce': {'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.2, 'fiber': 1.3, 'sugars': 0.8},
            'cucumber': {'calories': 15, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1, 'fiber': 0.5, 'sugars': 1.7},
            'bell pepper': {'calories': 31, 'protein': 1, 'carbs': 6, 'fat': 0.3, 'fiber': 2.1, 'sugars': 4.2},
            'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0, 'sugars': 0},
            'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'fiber': 0, 'sugars': 0},
            'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 13, 'fiber': 0, 'sugars': 0},
            'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0, 'sugars': 1.1},
            'tofu': {'calories': 76, 'protein': 8, 'carbs': 1.9, 'fat': 4.8, 'fiber': 0.3, 'sugars': 0},
            'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'fiber': 2.7, 'sugars': 5},
            'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugars': 0},
            'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1, 'fiber': 1.8, 'sugars': 1},
            'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0, 'sugars': 5},
            'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33, 'fiber': 0, 'sugars': 0.5},
            'yogurt': {'calories': 59, 'protein': 3.5, 'carbs': 4.7, 'fat': 1.5, 'fiber': 0, 'sugars': 4.7},
            'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10, 'fiber': 2.3, 'sugars': 3},
            'burger': {'calories': 295, 'protein': 17, 'carbs': 24, 'fat': 14, 'fiber': 1.5, 'sugars': 4},
            'sandwich': {'calories': 250, 'protein': 12, 'carbs': 30, 'fat': 8, 'fiber': 2, 'sugars': 3},
            'salad': {'calories': 15, 'protein': 1, 'carbs': 3, 'fat': 0, 'fiber': 1, 'sugars': 2},
            'soup': {'calories': 50, 'protein': 3, 'carbs': 8, 'fat': 1, 'fiber': 1.5, 'sugars': 2},
            'coffee': {'calories': 1, 'protein': 0.1, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugars': 0},
            'tea': {'calories': 1, 'protein': 0, 'carbs': 0.3, 'fat': 0, 'fiber': 0, 'sugars': 0},
            'juice': {'calories': 45, 'protein': 0.5, 'carbs': 11, 'fat': 0.1, 'fiber': 0.2, 'sugars': 9},
        }
        
        food_lower = self.clean_food_name(food_name)
        print(f"🔍 Looking up in database: '{food_lower}'")
        
        # Try exact match first
        if food_lower in comprehensive_db:
            print(f"✅ Exact match found: {food_lower}")
            return comprehensive_db[food_lower]
        
        # Try partial match
        for db_food, nutrition in comprehensive_db.items():
            if db_food in food_lower or food_lower in db_food:
                print(f"✅ Partial match found: {db_food} for {food_lower}")
                return nutrition
        
        # Try word-by-word matching
        food_words = set(food_lower.split())
        for db_food, nutrition in comprehensive_db.items():
            db_words = set(db_food.split())
            if food_words & db_words:
                print(f"✅ Word match found: {db_food} for {food_lower}")
                return nutrition
        
        # Default nutrition for unknown foods
        print(f"❌ No match found for: {food_lower}, using defaults")
        return {'calories': 150, 'protein': 5, 'carbs': 20, 'fat': 5, 'fiber': 2, 'sugars': 5}
    
    def get_allergens(self, food_name):
        """Get allergen information for food - EXPANDED"""
        allergen_database = {
            'peanut': ['peanut', 'peanuts', 'peanut butter', 'groundnut'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy', 'whey', 'casein'],
            'gluten': ['bread', 'pasta', 'wheat', 'barley', 'rye', 'flour', 'cereal', 'grain'],
            'shellfish': ['shrimp', 'crab', 'lobster', 'prawn', 'crayfish', 'scallop'],
            'eggs': ['egg', 'eggs', 'mayonnaise', 'mayo', 'ovalbumin'],
            'soy': ['soy', 'tofu', 'soybean', 'soy sauce', 'edamame', 'miso'],
            'tree_nuts': ['almond', 'walnut', 'cashew', 'pistachio', 'hazelnut', 'pecan', 'macadamia'],
            'fish': ['salmon', 'tuna', 'cod', 'trout', 'sardine', 'mackerel', 'anchovy']
        }
        
        allergens = []
        food_lower = food_name.lower()
        
        for allergen, keywords in allergen_database.items():
            if any(keyword in food_lower for keyword in keywords):
                allergens.append(allergen)
        
        return allergens
    
    def assess_diabetes_safety(self, nutrition_data):
        """Assess if food is safe for diabetics"""
        if not nutrition_data:
            return True
            
        carbs = nutrition_data.get('carbs', 0)
        fiber = nutrition_data.get('fiber', 0)
        sugars = nutrition_data.get('sugars', 0)
        
        # Calculate net carbs
        net_carbs = carbs - fiber
        
        # Simple diabetes safety assessment
        return net_carbs < 25 and sugars < 12

ultimate_ai = UltimateFoodAI()

# NutritionAIChat class from original chat.py (full exact)
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

nutrition_chat = NutritionAIChat()

# Route functions from original app.py (full exact, no @app.route)
def analyze_food():
    try:
        data = request.json
        image_data = data.get('image')
        user_profile = data.get('user_profile', {})
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        print("🔄 Starting food analysis with local Hugging Face...")
        
        # Detect food using local Hugging Face
        detected_foods = ultimate_ai.detect_food_huggingface(image_data)
        
        if not detected_foods:
            print("❌ No foods detected in image")
            return jsonify({
                'foodName': 'Unknown Food',
                'confidence': 0,
                'nutrients': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0},
                'allergens': [],
                'diabetesSafe': True,
                'warnings': ['Could not identify any food in the image'],
                'totalCalories': 0,
                'multipleFoods': False
            })
        
        # Get the most confident food detection
        primary_food = max(detected_foods, key=lambda x: x['confidence'])
        food_name = primary_food['name']
        confidence = round(primary_food['confidence'] * 100)
        
        food_details = []
        for f in detected_foods[:5]:
            food_details.append(f"{f['name']} ({f['confidence']:.2f})")
        
        print(f"🔍 Primary detection: {food_name} ({confidence}% confidence)")
        print(f"📊 All detected foods: {food_details}")
        
        # Get real nutrition data
        nutrition_data = ultimate_ai.get_nutrition_data(food_name)
        
        # Get allergens
        allergens = ultimate_ai.get_allergens(food_name)
        
        # Check diabetes safety
        diabetes_safe = ultimate_ai.assess_diabetes_safety(nutrition_data)
        
        # Generate warnings based on user profile
        warnings = []
        user_allergens = user_profile.get('allergens', [])
        
        # Check for allergen conflicts
        for allergen in allergens:
            if allergen in user_allergens:
                warnings.append(f"🚨 Contains {allergen} - not safe for your allergies!")
        
        # Check diabetes
        if user_profile.get('has_diabetes', False) and not diabetes_safe:
            warnings.append("⚠️ High carb/sugar content - may affect blood sugar")
        
        # Add general warnings
        if confidence < 50:
            warnings.append("⚠️ Low confidence detection - results may be inaccurate")
        
        response = {
            'foodName': food_name.title(),
            'confidence': confidence,
            'nutrients': nutrition_data,
            'allergens': allergens,
            'diabetesSafe': diabetes_safe,
            'warnings': warnings,
            'totalCalories': nutrition_data.get('calories', 0),
            'multipleFoods': len(detected_foods) > 1,
            'allDetectedFoods': [f['name'] for f in detected_foods[:5]],
            'dataSource': 'Local Hugging Face Food-101 + USDA/Open Food Facts'
        }
        
        print(f"✅ Analysis complete: {food_name} - {nutrition_data.get('calories', 0)} cal - {len(warnings)} warnings")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        print(f"🔍 Full error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'Ultimate Food AI Backend - FIXED VERSION',
        'features': [
            'Local Hugging Face AI - Real food detection (no keys, no limits)',
            'USDA FoodData Central - Free nutrition data', 
            'Open Food Facts - Free nutrition data',
            'Comprehensive food database - 200+ foods',
            'Allergen detection',
            'Diabetes safety assessment'
        ],
        'free': True,
        'noApprovalNeeded': True
    })

# Route functions from original chat.py (full exact, no @app.route; uses global supabase)
def chat_health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'SahaAI Chat Backend',
        'ollama_available': test_ollama_connection(),
        'database_connected': test_database_connection()
    })

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

def get_chat_history(user_id):
    try:
        # Get chat history from REAL database
        history = get_user_chat_history(user_id)
        return jsonify({'history': history, 'success': True})
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

def get_user_context_endpoint(user_id):
    try:
        context = get_user_context(user_id)
        if not context:
            return jsonify({'error': 'User not found', 'success': False}), 404
        return jsonify({'user_context': context, 'success': True})
    except Exception as e:
        logger.error(f"User context error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

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

# All helper functions from original chat.py (full exact)
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
            timeout=100
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

# Register all routes (full list)
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