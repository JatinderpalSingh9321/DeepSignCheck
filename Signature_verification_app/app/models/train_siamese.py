import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.siamese_model import train_siamese_model

if __name__ == "__main__":
    dataset_path = "dataset/train"  # Folder with all author subfolders
    model_output_path = "app/models/siamese_model.pth"
    num_epochs = 20

    train_siamese_model(dataset_path, num_epochs=num_epochs, model_output_path=model_output_path)
