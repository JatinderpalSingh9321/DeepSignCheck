import os
import shutil
import torch
import io
from flask import Blueprint, render_template, request, url_for, current_app
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from pymongo import MongoClient
import gridfs
from app.models.siamese_model import load_model
import torch.nn.functional as F

main = Blueprint('main', __name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "siamese_model.pth")
model = load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["signatureDB"]
fs = gridfs.GridFS(db)

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# Function to identify the best-matching image for the selected person
def identify_person(uploaded_img, selected_person, threshold=0.5):
    uploaded_img_tensor = transform(uploaded_img).unsqueeze(0).to(device)

    best_distance = float("inf")
    best_image_data = None
    best_filename = None

    # Query only genuine signatures of the selected person
    genuine_signatures = fs.find({"person_id": selected_person, "type": "genuine"})

    for file in genuine_signatures:
        try:
            ref_img = Image.open(io.BytesIO(file.read())).convert("L")
            ref_tensor = transform(ref_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output1, output2 = model(uploaded_img_tensor, ref_tensor)
                distance = F.pairwise_distance(output1, output2).item()

            print(f"[DEBUG] Distance to {selected_person}/{file.filename}: {distance:.4f}")

            if distance < best_distance:
                best_distance = distance
                best_image_data = ref_img
                best_filename = file.filename
        except Exception as e:
            print(f"[ERROR] Skipping image: {e}")

    if best_image_data is None:
        return f"❌ No reference images found for {selected_person}.", None

    # Save best match image for display
    upload_folder = current_app.config['UPLOAD_FOLDER']
    matched_filename = f"matched_{best_filename}"
    matched_path = os.path.join(upload_folder, matched_filename)
    best_image_data.save(matched_path)
    matched_image_url = f"uploads/{matched_filename}"

    if best_distance < threshold:
        return f"✅ Genuine signature of {selected_person} (distance = {best_distance:.4f})", matched_image_url
    else:
        return f"❌ Signature does not match {selected_person} (distance = {best_distance:.4f})", matched_image_url

# Flask route
@main.route('/', methods=['GET', 'POST'])
def index():
    # Fetch all unique person IDs from MongoDB
    people = db.fs.files.distinct("person_id", {"type": "genuine"})
    result = None
    uploaded_image_url = None
    matched_image_url = None

    if request.method == 'POST':
        person = request.form['person']
        file = request.files['signature']
        if file:
            filename = secure_filename(file.filename)

            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)

            upload_path = os.path.join(upload_folder, filename)
            file.save(upload_path)

            uploaded_img = Image.open(upload_path).convert("L")
            result, matched_image_url = identify_person(uploaded_img, person)

            uploaded_image_url = f"uploads/{filename}"

    return render_template("index.html",
                           people=people,
                           result=result,
                           uploaded_image=uploaded_image_url,
                           matched_image=matched_image_url)
