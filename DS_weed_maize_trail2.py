import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Progress bar

# Configuration
NUM_CLASSES = 2
MAX_BOXES = 20
IMAGE_SIZE = 640

# Custom Dataset Class
class AgricultureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("Initializing Dataset...")
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        with open(os.path.join(self.root_dir, f'{base_name}.json')) as f:
            data = json.load(f)
        
        boxes, labels = [], []
        for ann in data[0]['annotations']:
            x, y, w, h = ann['coordinates'].values()
            boxes.append([x / IMAGE_SIZE, y / IMAGE_SIZE, (x + w) / IMAGE_SIZE, (y + h) / IMAGE_SIZE])
            labels.append(0 if ann['label'] == 'maize' else 1)
        
        while len(boxes) < MAX_BOXES:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        return image, {'boxes': boxes, 'labels': labels}

# CNN Model
class ObjectDetectionCNN(nn.Module):
    def __init__(self):
        super(ObjectDetectionCNN, self).__init__()
        print("Initializing Model...")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 80 * 80, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, MAX_BOXES * (4 + NUM_CLASSES))
        )
    
    def forward(self, x):
        return self.detector(self.features(x)).view(-1, MAX_BOXES, 4 + NUM_CLASSES)

# Loss Function
class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        print("Initializing Loss Function...")
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, outputs, targets):
        pred_boxes, pred_classes = outputs[..., :4], outputs[..., 4:]
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        box_loss = self.bbox_loss(pred_boxes, gt_boxes)
        cls_loss = self.cls_loss(pred_classes.view(-1, NUM_CLASSES), gt_labels.view(-1))
        return box_loss + cls_loss

# Training Function
def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    print("Starting Training...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss/len(dataloader):.4f}")
        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")

# Main Execution
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset = AgricultureDataset('Annotated-Maize-Weed-Images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ObjectDetectionCNN()
    criterion = DetectionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(dataloader, model, criterion, optimizer, num_epochs=20)
    print("Training Complete!")