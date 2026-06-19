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

# Configuration
NUM_CLASSES = 2  # maize and weed
MAX_BOXES = 20  # Maximum number of boxes per image
IMAGE_SIZE = 640

# Custom Dataset Class
class AgricultureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        with open(os.path.join(self.root_dir, f'{base_name}.json')) as f:
            data = json.load(f)
        
        # Initialize targets
        boxes = []
        labels = []
        
        # Process annotations
        for ann in data[0]['annotations']:
            # Convert coordinates to [x_min, y_min, x_max, y_max]
            x = ann['coordinates']['x']
            y = ann['coordinates']['y']
            w = ann['coordinates']['width']
            h = ann['coordinates']['height']
            
            boxes.append([
                x / IMAGE_SIZE,          # x_min (normalized)
                y / IMAGE_SIZE,          # y_min (normalized)
                (x + w) / IMAGE_SIZE,    # x_max (normalized)
                (y + h) / IMAGE_SIZE,    # y_max (normalized)
            ])
            labels.append(0 if ann['label'] == 'maize' else 1)
        
        # Pad to MAX_BOXES
        while len(boxes) < MAX_BOXES:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)  # Invalid label

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, {'boxes': boxes, 'labels': labels}

# CNN Architecture
class ObjectDetectionCNN(nn.Module):
    def __init__(self):
        super(ObjectDetectionCNN, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 80 * 80, 512),  # Adjust based on input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, MAX_BOXES * (4 + NUM_CLASSES))  # 4 coordinates + class scores per box
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.view(-1, MAX_BOXES, 4 + NUM_CLASSES)

# Custom Loss Function
class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, targets):
        # Split outputs
        pred_boxes = outputs[..., :4]
        pred_classes = outputs[..., 4:]
        
        # Get targets
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        
        # Calculate losses
        box_loss = self.bbox_loss(pred_boxes, gt_boxes)
        cls_loss = self.cls_loss(pred_classes.view(-1, NUM_CLASSES), gt_labels.view(-1))
        
        return box_loss + cls_loss

# Training Function
def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}')

# Visualization Function
def visualize_predictions(image, predictions, confidence=0.5):
    draw = ImageDraw.Draw(image)
    for box, cls in zip(predictions['boxes'], predictions['classes']):
        if cls != -1:
            # Convert normalized coordinates to image size
            x_min = box[0] * IMAGE_SIZE
            y_min = box[1] * IMAGE_SIZE
            x_max = box[2] * IMAGE_SIZE
            y_max = box[3] * IMAGE_SIZE
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
            draw.text((x_min, y_min), f"{'maize' if cls == 0 else 'weed'}", fill='red')
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset = AgricultureDataset('Annotated-Maize-Weed-Images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = ObjectDetectionCNN()
    criterion = DetectionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(dataloader, model, criterion, optimizer, num_epochs=20)

    # Test prediction
    sample_image, sample_target = dataset[0]
    with torch.no_grad():
        outputs = model(sample_image.unsqueeze(0))
    
    # Process outputs
    boxes = outputs[0, :, :4].cpu().numpy()
    classes = outputs[0, :, 4:].argmax(dim=-1).cpu().numpy()
    
    # Visualize
    original_image = Image.open(os.path.join('Annotated-Maize-Weed-Images', dataset.image_files[0]))
    visualize_predictions(original_image, {
        'boxes': boxes,
        'classes': classes
    })