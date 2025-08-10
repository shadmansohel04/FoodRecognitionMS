from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.datasets import Food101
import torch.nn.functional as F
import io
import math

app = Flask(__name__)
CORS(app)
CONF_THRESHOLD = 0.7

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 101) 
model.load_state_dict(torch.load("food101_resnet50.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = Food101(root='data', download=True).classes

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'frame' not in request.files:
            raise Exception("no file")

        file = request.files['frame']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)
            max_prob = max_prob.item()

        if max_prob < CONF_THRESHOLD:
            return jsonify({
                "success": False,
                "class": None,
                "confidence": max_prob
            })
        
        return jsonify({
            "success": True,
            "class": class_names[predicted_class.item()].replace("_", " "),
            "confidence": math.ceil(max_prob * 100)
        })

    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e)}
        ), 400
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)