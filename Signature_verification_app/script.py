from pymongo import MongoClient
import gridfs
import os

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["signatureDB"]
fs = gridfs.GridFS(db)

# Dataset root (change if needed)
DATASET_PATH = "dataset"

# Traverse all person folders
for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    
    if not os.path.isdir(person_dir):
        continue  # Skip any non-folder entries

    for category in ["genuine", "forged"]:
        category_dir = os.path.join(person_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for fname in os.listdir(category_dir):
            if not fname.lower().endswith(".png"):
                continue

            file_path = os.path.join(category_dir, fname)

            # Check if this file already exists (by filename + person + category)
            exists = fs.find_one({
                "filename": fname,
                "person_id": person,
                "type": category
            })

            if exists:
                print(f"[SKIP] Already exists: {person}/{category}/{fname}")
                continue

            with open(file_path, "rb") as f:
                file_data = f.read()

            fs.put(file_data,
                   filename=fname,
                   person_id=person,
                   type=category)

            print(f"[INSERTED] {person}/{category}/{fname}")
