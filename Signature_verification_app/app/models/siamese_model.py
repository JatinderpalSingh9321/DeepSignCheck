import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# Siamese Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, 1, 105, 105)
        dummy_output = self.cnn(dummy_input)
        self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# CEDAR Dataset
class CEDARDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.labels = []

        print(f"[INFO] Loading dataset from: {root_dir}")
        people = os.listdir(root_dir)

        for person in people:
            person_dir = os.path.join(root_dir, person)
            genuine_dir = os.path.join(person_dir, "genuine")
            forged_dir = os.path.join(person_dir, "forged")

            if not os.path.isdir(genuine_dir) or not os.path.isdir(forged_dir):
                print(f"[WARN] Skipping '{person}' — missing 'genuine/' or 'forged/' folder.")
                continue

            print(f"[INFO] Processing person folder: {person}")

            genuine_imgs = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) if f.endswith(".png")]
            forged_imgs = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir) if f.endswith(".png")]

            # Create genuine-genuine pairs (label 0)
            for i in range(len(genuine_imgs)):
                for j in range(i + 1, len(genuine_imgs)):
                    self.pairs.append((genuine_imgs[i], genuine_imgs[j]))
                    self.labels.append(0)

            # Create genuine-forged pairs (label 1)
            for i in range(min(len(genuine_imgs), len(forged_imgs))):
                self.pairs.append((genuine_imgs[i], forged_imgs[i]))
                self.labels.append(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Training function
def train_siamese_model(dataset_path, num_epochs=20, batch_size=16, learning_rate=0.001, model_output_path="siamese_model.pth"):
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    dataset = CEDARDataset(dataset_path, transform)

    # 80/10/10 split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("[INFO] Starting training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
        print(f"           Validation Loss: {val_loss:.4f}")

    # Final Test Evaluation
    model.eval()
    test_loss = 0
    all_labels = []
    all_predictions = []
    threshold = 0.5  # tune this threshold as needed

    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            test_loss += loss.item()

            distance = F.pairwise_distance(output1, output2)
            prediction = (distance > threshold).float()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(prediction.cpu().numpy())

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"[INFO] Final Test Loss: {test_loss:.4f}")
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")
    print(f"[INFO] Test Precision: {precision:.4f}")
    print(f"[INFO] Test Recall: {recall:.4f}")
    print(f"[INFO] Test F1-score: {f1:.4f}")


   # ✅ Ensure the directory exists before saving the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # ✅ Save the model
    torch.save(model.state_dict(), model_output_path)
    print(f"[INFO] Model saved to {model_output_path}")
    print(f"[DEBUG] Model absolute path: {os.path.abspath(model_output_path)}")


# Load model function
def load_model(model_path="siamese_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
