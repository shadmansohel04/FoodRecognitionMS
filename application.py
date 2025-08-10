from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import io
import math
import gc
import os

application = Flask(__name__)
CORS(application)
CONF_THRESHOLD = 0.7

# Device (force CPU if you want predictable memory usage)
device = torch.device("cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 101)
state_dict = torch.load("food101_resnet50.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class_names = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake",
    "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
    "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
    "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
    "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@application.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": "yipppee"
    })

@application.route('/detect', methods=['POST'])
def detect():
    try:
        if 'frame' not in request.files:
            raise Exception("no file")

        file = request.files['frame']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)

        max_prob = float(max_prob.item())
        pred_idx = predicted_class.item()

        # Free memory early
        del input_tensor, outputs, probabilities, predicted_class
        gc.collect()

        if max_prob < CONF_THRESHOLD:
            return jsonify({
                "success": False,
                "class": None,
                "confidence": max_prob
            })

        return jsonify({
            "success": True,
            "class": class_names[pred_idx].replace("_", " "),
            "confidence": math.ceil(max_prob * 100)
        })

    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 400

if __name__ == '__main__':
    port = os.getenv("PORT") or 3000
    application.run(debug=True, port=port)
