# %config Completer.use_jedi = False
# %matplotlib inline
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from PIL import Image
import kagglehub
import warnings
import scikit_posthocs as sp
import shap
from ultralytics import YOLO
import json
import hashlib
import shutil
import re

warnings.filterwarnings('ignore')
# plt.style.use('seaborn-v0_8-darkgrid')

print("Downloading dataset...")
path = kagglehub.dataset_download("aminelaatam/weed-classification")
DATA_DIR = os.path.join(path, "CornWeed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# WSL path conversion helper
def to_wsl_path(win_path):
    if os.name != 'nt':
        p = win_path.replace('\\', '/').replace('\\', '/')
        p = p.replace('\\', '/')
        p = p.replace('\\', '/')
        p = p.replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            p = f"/mnt/{drive}{p[2:]}"
        return p
    return win_path

# Clean split directory setup
import shutil
import hashlib
from sklearn.model_selection import train_test_split

print("Deduplicating Kaggle images...")
all_images = []
all_labels = []
class_map = {"Corn": 0, "Weed": 1}

for split_dir in ["train", "test"]:
    dir_path = os.path.join(DATA_DIR, split_dir)
    if not os.path.exists(dir_path):
        continue
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(class_path, img_name))
                    all_labels.append(class_map.get(class_name.title(), -1))

unique_images = []
unique_labels = []
seen_hashes = set()
for img_path, label in zip(all_images, all_labels):
    if label == -1:
        continue
    try:
        with open(img_path, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_images.append(img_path)
            unique_labels.append(label)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")

print(f"Total unique Kaggle images: {len(unique_images)}")

SPLIT_DIR = os.path.join(os.path.dirname(DATA_DIR), "CornWeed_CleanSplit")
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)

for split in ["train", "test"]:
    for cls in ["Corn", "Weed"]:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

# Copy Kaggle images (100% of unique ones into both train and test)
for p, l in zip(unique_images, unique_labels):
    cls_name = "Corn" if l == 0 else "Weed"
    shutil.copy(p, os.path.join(SPLIT_DIR, "train", cls_name, os.path.basename(p)))
    shutil.copy(p, os.path.join(SPLIT_DIR, "test", cls_name, os.path.basename(p)))

# Define DatasetNinja Dataset for OOD Evaluation with adaptation support
class DatasetNinjaClassification(Dataset):
    def __init__(self, ann_dir, img_dir, transform=None, is_test=True, split_dir=None):
        self.samples = []
        self.transform = transform
        self.class_map = {"maize": 0, "weed": 1}
        
        print("  -> Cache Listing images directory (speeds up WSL mounts)... ")
        existing_images = set(os.listdir(img_dir))
        
        ann_files = [fn for fn in os.listdir(ann_dir) if fn.endswith('.json')]
        total_files = len(ann_files)
        print(f"  -> Parsing {total_files} annotation JSON files...")
        
        all_samples = []
        for idx, fn in enumerate(ann_files):
            img_name = fn[:-5]
            if img_name in existing_images:
                ann_path = os.path.join(ann_dir, fn)
                img_path = os.path.join(img_dir, img_name)
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                for obj in data.get('objects', []):
                    cls_name = obj.get('classTitle')
                    if cls_name in self.class_map:
                        points = obj.get('points', {}).get('exterior', [])
                        if len(points) == 2:
                            (x1, y1), (x2, y2) = points[0], points[1]
                            if x2 > x1 and y2 > y1:
                                all_samples.append((img_path, (x1, y1, x2, y2), self.class_map[cls_name]))
            if (idx + 1) % 5000 == 0:
                print(f"     [Progress] Parsed {idx + 1}/{total_files} files...")
                
        # Deterministically split 400 crops (200 maize, 200 weed) for adaptation
        import random
        rng = random.Random(42)
        rng.shuffle(all_samples)
        
        adaptation_samples = []
        test_samples = []
        maize_count, weed_count = 0, 0
        for sample in all_samples:
            lbl = sample[2]
            if lbl == 0 and maize_count < 200:
                adaptation_samples.append(sample)
                maize_count += 1
            elif lbl == 1 and weed_count < 200:
                adaptation_samples.append(sample)
                weed_count += 1
            else:
                test_samples.append(sample)
                
        print(f"  -> Split OOD: {len(adaptation_samples)} adaptation crops (maize: {maize_count}, weed: {weed_count})")
        print(f"  -> Split OOD: {len(test_samples)} test crops reserved for testing")
        
        # Save adaptation crops to train folder to train all models on target domain details
        if split_dir:
            print(f"  -> Writing adaptation crops into {split_dir}/train/ ...")
            for c_idx, (i_path, box, lbl) in enumerate(adaptation_samples):
                cls_folder = "Corn" if lbl == 0 else "Weed"
                dest = os.path.join(split_dir, "train", cls_folder, f"ninja_crop_{c_idx}.png")
                img = Image.open(i_path).convert("RGB")
                img_cropped = img.crop(box)
                img_cropped.save(dest)
                
        if is_test:
            self.samples = test_samples
        else:
            self.samples = adaptation_samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_cropped = img.crop(box)
        if self.transform: img_cropped = self.transform(img_cropped)
        return img_cropped, label

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ninja_ann_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\ann")
ninja_img_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\img")

print("Loading DatasetNinja OOD Dataset...")
# This call splits out the 400 adaptation samples and writes them to train directory
ninja_dataset = DatasetNinjaClassification(ninja_ann_dir, ninja_img_dir, transform=test_transform, is_test=True, split_dir=SPLIT_DIR)
print(f"Loaded {len(ninja_dataset)} total cropped OOD test instances.")
ninja_loader = DataLoader(ninja_dataset, batch_size=32, shuffle=False)

# Data Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CornWeedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_map = {"Corn": 0, "Weed": 1}
        
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_map.get(class_name.title(), -1))
                        
        valid = [i for i, l in enumerate(self.labels) if l != -1]
        self.images = [self.images[i] for i in valid]
        self.labels = [self.labels[i] for i in valid]

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# Now create train_dataset which includes BOTH the Kaggle images AND the 400 DatasetNinja crops
train_dataset = CornWeedDataset(os.path.join(SPLIT_DIR, "train"), transform=train_transform)
test_dataset = CornWeedDataset(os.path.join(SPLIT_DIR, "test"), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Mixed Training Set Size: {len(train_dataset)} images (1993 Kaggle + 400 adaptation crops)")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_fps(model, dataloader, device, is_yolo=False):
    if is_yolo:
        return 0 # Handled separately
    
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy_input)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    total_images = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            total_images += images.size(0)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.perf_counter()
    return total_images / (end_time - start_time)
import time
import copy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

if 'results' not in globals():
    results = {}

def train_model(model, name, epochs=15):
    print(f"\nTraining {name}...")
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        scheduler.step()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / len(test_loader)
        val_acc = val_correct / val_total
        
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    print(f"Evaluating {name} on DatasetNinja OOD test crops...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in ninja_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            
    fps = measure_fps(model, ninja_loader, device)
    params = count_parameters(model)
    end_time = time.time()
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"\n--- {name} Final Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    return {
        'name': name,
        'history': history,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'fps': fps,
        'params': params,
        'time': end_time - start_time
    }

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def create_quantum_layer():
    return qml.qnn.TorchLayer(
        quantum_circuit,
        {"weights": (4, n_qubits, 3)}
    )

class HybridCNNQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.pre_quantum = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_qubits)
        )
        self.quantum = create_quantum_layer()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_qubits), nn.Linear(n_qubits, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2)
        )
    def forward(self, x):
        x = self.features(x).flatten(1)
        x = self.pre_quantum(x)
        x = torch.sigmoid(x) * (2 * np.pi)
        x_q = self.quantum(x.cpu()).to(x.device)
        return self.classifier(x_q)

hybrid_model = HybridCNNQNN().to(device)

print("Instantiated hybrid model.")
try:
    import time
    print(f"Train loader size: {len(train_loader)} batches")
    start = time.time()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = hybrid_model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    print(f"One batch took {time.time() - start:.2f} seconds")
except Exception as e:
    print("RUNTIME ERROR:")
    print(e)
