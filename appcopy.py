import os
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Load the model
def load_model():
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load("best_bloodgroup_densenet121.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        return class_names[pred.item()], confidence.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, confidence, message, img_data = None, None, None, None

    if request.method == 'POST':
        if 'file' not in request.files:
            message = "No file part in request."
            return render_template('index.html', message=message)

        file = request.files['file']
        if file.filename == '':
            message = "No file selected."
        elif not allowed_file(file.filename):
            message = "Only .bmp files are allowed!"
        else:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                prediction, confidence = predict_image(filepath)
                confidence_percent = round(confidence * 100, 2)

                # Confidence threshold check
                if confidence_percent < 40:
                    prediction = None
                    message = "Unable to predict due to low accuracy."
                else:
                    # Convert image to base64 for display
                    with open(filepath, "rb") as image_file:
                        img_data = base64.b64encode(image_file.read()).decode("utf-8")

            except Exception as e:
                message = f"Prediction error: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=round(confidence * 100, 2) if prediction else None,
        message=message,
        img_data=img_data
    )

if __name__ == '__main__':
    app.run(debug=True)
