import os
import time
import json
import hashlib
import shutil
import copy
import warnings
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models

import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import kagglehub
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ==============================================================================
# Helper Functions and Dataset Definitions
# ==============================================================================
def to_wsl_path(win_path):
    if os.name != 'nt':
        p = win_path.replace('\\', '/').replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            p = f"/mnt/{drive}{p[2:]}"
        return p
    return win_path

class CornWeedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = [], []
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

class DatasetNinjaClassification(Dataset):
    def __init__(self, ann_dir, img_dir, transform=None, is_test=True, split_dir=None):
        self.samples = []
        self.transform = transform
        self.class_map = {"maize": 0, "weed": 1}
        
        existing_images = set(os.listdir(img_dir))
        ann_files = [fn for fn in os.listdir(ann_dir) if fn.endswith('.json')]
        
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
                                
        import random
        rng = random.Random(42)
        rng.shuffle(all_samples)
        
        adaptation_samples, test_samples = [], []
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
                
        if split_dir:
            for c_idx, (i_path, box, lbl) in enumerate(adaptation_samples):
                cls_folder = "Corn" if lbl == 0 else "Weed"
                dest = os.path.join(split_dir, "train", cls_folder, f"ninja_crop_{c_idx}.png")
                img = Image.open(i_path).convert("RGB")
                img.crop(box).save(dest)
                
        self.samples = test_samples if is_test else adaptation_samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB").crop(box)
        if self.transform: img = self.transform(img)
        return img, label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================================
# Model Definitions
# ==============================================================================
# --- Legacy CNN ---
class Custom_Legacy_CNN(nn.Module):
    def __init__(self):
        super(Custom_Legacy_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- Quantum Setup ---
n_qubits = 10
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def upgraded_quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def create_upgraded_quantum_layer():
    return qml.qnn.TorchLayer(upgraded_quantum_circuit, {"weights": (2, n_qubits, 3)})

# --- Hybrid Upgraded Models ---
class HybridCNNQNN_Upgraded(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = nn.Linear(128, 1024)
        self.quantum = create_upgraded_quantum_layer()
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        x_q = self.quantum(x.cpu()).to(x.device)
        return self.fc2(x_q)

class Custom_Legacy_QNN_Upgraded(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 1024)
        self.quantum = create_upgraded_quantum_layer()
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        x_q = self.quantum(x.cpu()).to(x.device)
        return self.fc2(x_q)

class HybridCNNQNN_Transfer_Upgraded(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        for param in self.features.parameters(): param.requires_grad = False
        self.fc1 = nn.Linear(1280, 1024)
        self.quantum = create_upgraded_quantum_layer()
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        x_q = self.quantum(x.cpu()).to(x.device)
        return self.fc2(x_q)

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing Benchmark Pipeline on {device} ---")
    
    # 1. Dataset Setup
    path = kagglehub.dataset_download("aminelaatam/weed-classification")
    DATA_DIR = os.path.join(path, "CornWeed")
    SPLIT_DIR = os.path.join(os.path.dirname(DATA_DIR), "CornWeed_CleanSplit")
    
    # Setup data split directory if it doesn't exist
    if not os.path.exists(SPLIT_DIR):
        print("Building clean dataset split...")
        os.makedirs(SPLIT_DIR)
        for split in ["train", "test"]:
            for cls in ["Corn", "Weed"]:
                os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)
                
        # Deduplicate Kaggle and copy
        seen_hashes = set()
        for split_dir in ["train", "test"]:
            dir_path = os.path.join(DATA_DIR, split_dir)
            if not os.path.exists(dir_path): continue
            for class_name in os.listdir(dir_path):
                class_path = os.path.join(dir_path, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            with open(img_path, 'rb') as f: h = hashlib.md5(f.read()).hexdigest()
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                cls = "Corn" if class_name.title() == "Corn" else "Weed"
                                shutil.copy(img_path, os.path.join(SPLIT_DIR, "train", cls, img_name))
                                shutil.copy(img_path, os.path.join(SPLIT_DIR, "test", cls, img_name))
                        except Exception: pass

    # Prepare Ninja OOD Dataset
    ninja_ann_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\ann")
    ninja_img_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\img")
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading OOD and Train Data...")
    ninja_dataset = DatasetNinjaClassification(ninja_ann_dir, ninja_img_dir, transform=test_transform, is_test=True, split_dir=SPLIT_DIR)
    ninja_loader = DataLoader(ninja_dataset, batch_size=32, shuffle=False)

    train_dataset = CornWeedDataset(os.path.join(SPLIT_DIR, "train"), transform=train_transform)
    test_dataset = CornWeedDataset(os.path.join(SPLIT_DIR, "test"), transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Setup Saved Models Directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    results_summary = []
    results_raw = {} # Store raw arrays for plotting

    # Helper Training Function
    def train_and_evaluate(model, name, epochs=15):
        print(f"\n=============================================")
        print(f"🚀 Training {name}")
        print(f"=============================================")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())
        
        history = {'acc': []}
        
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images.to(device))
                    _, pred = torch.max(outputs, 1)
                    val_correct += (pred == labels.to(device)).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            history['acc'].append(val_acc)
            print(f"  Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())
        
        # Load best and save
        model.load_state_dict(best_wts)
        model_path = os.path.join(save_dir, f"{name.replace(' ', '_')}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Evaluate on OOD
        print(f"📊 Evaluating {name} on OOD DatasetNinja...")
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        start_time = time.time()
        with torch.no_grad():
            for images, labels in ninja_loader:
                outputs = model(images.to(device))
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        fps = len(all_labels) / (time.time() - start_time)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        params = count_parameters(model)
        
        print(f"  --> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPS: {fps:.1f}")
        
        results_summary.append({
            "Model": name, "OOD_Accuracy": acc, "OOD_F1": f1, "OOD_Precision": prec, 
            "OOD_Recall": rec, "OOD_AUC": auc, "FPS": fps, "Params": params
        })
        results_raw[name] = {
            'labels': np.array(all_labels),
            'preds': np.array(all_preds),
            'probs': np.array(all_probs),
            'history': history,
            'fps': fps,
            'params': params
        }

    # 3. Model Initialization
    resnet = models.resnet18(weights='DEFAULT')
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    
    mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)
    
    efficientnet = models.efficientnet_b0(weights='DEFAULT')
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)

    models_to_run = [
        (resnet.to(device), "ResNet18"),
        (mobilenet.to(device), "MobileNetV3"),
        (efficientnet.to(device), "EfficientNet-B0"),
        (Custom_Legacy_CNN().to(device), "Custom_Legacy_CNN"),
        (HybridCNNQNN_Upgraded().to(device), "Hybrid CNN-QNN (From Scratch)"),
        (Custom_Legacy_QNN_Upgraded().to(device), "Custom_Legacy_QNN"),
        (HybridCNNQNN_Transfer_Upgraded().to(device), "Hybrid CNN-QNN (Transfer Learning)")
    ]

    for m, name in models_to_run:
        train_and_evaluate(m, name, epochs=15)

    # 4. YOLOv8
    print(f"\n=============================================")
    print(f"🚀 Training YOLOv8n-cls")
    print(f"=============================================")
    yolo_model = YOLO('yolov8n-cls.pt')
    yolo_model.train(data=SPLIT_DIR, epochs=15, imgsz=128, batch=32, device='0' if torch.cuda.is_available() else 'cpu', verbose=False, workers=0)
    
    print(f"📊 Evaluating YOLOv8n-cls on OOD DatasetNinja...")
    yolo_preds, yolo_probs, yolo_labels = [], [], []
    t = time.time()
    for idx in range(len(ninja_dataset)):
        img_path, box, label = ninja_dataset.samples[idx]
        res = yolo_model(Image.open(img_path).convert("RGB").crop(box), verbose=False)[0]
        yolo_probs.append(res.probs.data[1].item())
        yolo_preds.append(res.probs.top1)
        yolo_labels.append(label)
    fps_yolo = len(ninja_dataset) / (time.time() - t)
    
    acc_y = accuracy_score(yolo_labels, yolo_preds)
    f1_y = f1_score(yolo_labels, yolo_preds, zero_division=0)
    print(f"  --> Acc: {acc_y:.4f} | F1: {f1_y:.4f} | FPS: {fps_yolo:.1f}")
    yolo_params = sum(p.numel() for p in yolo_model.parameters())
    results_summary.append({
        "Model": "YOLOv8n-cls", "OOD_Accuracy": acc_y, "OOD_F1": f1_y, "OOD_Precision": precision_score(yolo_labels, yolo_preds, zero_division=0),
        "OOD_Recall": recall_score(yolo_labels, yolo_preds, zero_division=0), "OOD_AUC": roc_auc_score(yolo_labels, yolo_probs), "FPS": fps_yolo, "Params": yolo_params
    })
    results_raw["YOLOv8n-cls"] = {
        'labels': np.array(yolo_labels), 'preds': np.array(yolo_preds), 'probs': np.array(yolo_probs),
        'history': {'acc': [0]*15}, 'fps': fps_yolo, 'params': yolo_params
    }

    # 5. QSVC Benchmark
    print(f"\n=============================================")
    print(f"🚀 Running QSVC Benchmark")
    print(f"=============================================")
    def extract_features(dataloader, model):
        model.eval()
        all_features, all_lbls = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                x = model.features(images.to(device))
                x = model.avgpool(x).view(x.size(0), -1)
                x = torch.nn.functional.normalize(model.fc1(x), p=2, dim=1, eps=1e-8)
                all_features.append(x.cpu().numpy())
                all_lbls.append(labels.numpy())
        return np.vstack(all_features), np.concatenate(all_lbls)

    tl_model = models_to_run[-1][0] # Hybrid CNN-QNN (Transfer Learning)
    train_features, train_labels_qsvc = extract_features(train_loader, tl_model)
    ninja_features, ninja_labels_qsvc = extract_features(ninja_loader, tl_model)

    subset_size = min(100, len(train_features))
    if len(np.unique(train_labels_qsvc)) > 1:
        train_features_sub, _, train_labels_sub, _ = train_test_split(
            train_features, train_labels_qsvc, train_size=subset_size, 
            stratify=train_labels_qsvc, random_state=42
        )
    else:
        train_features_sub = train_features[:subset_size]
        train_labels_sub = train_labels_qsvc[:subset_size]
    
    def compute_kernel_matrix(A, B):
        print(f"  -> Computing {len(A)}x{len(B)} kernel matrix (using classical algebraic equivalent)...")
        # Since the quantum kernel circuit is just AmplitudeEmbedding(x1) followed by adjoint(AmplitudeEmbedding(x2)),
        # it mathematically simplifies exactly to the squared dot product of the two L2-normalized vectors.
        return np.clip((A @ B.T) ** 2, 0, 1.0)

    print("Computing Kernel Matrix and Fitting SVC...")
    K_train = compute_kernel_matrix(train_features_sub, train_features_sub)
    svc = SVC(kernel="precomputed", probability=True)
    svc.fit(K_train, train_labels_sub)
    
    model_path = os.path.join(save_dir, "Hybrid_CNN_QSVC.pkl")
    joblib.dump(svc, model_path)
    print(f"✅ Model saved to {model_path}")
    
    print("Evaluating QSVC on OOD DatasetNinja...")
    K_test = compute_kernel_matrix(ninja_features, train_features_sub)
    qsvc_preds = svc.predict(K_test)
    qsvc_probs = svc.predict_proba(K_test)[:, 1]
    
    acc_q = accuracy_score(ninja_labels_qsvc, qsvc_preds)
    f1_q = f1_score(ninja_labels_qsvc, qsvc_preds, zero_division=0)
    print(f"  --> Acc: {acc_q:.4f} | F1: {f1_q:.4f}")
    
    try:
        auc_q = roc_auc_score(ninja_labels_qsvc, qsvc_probs)
    except ValueError:
        auc_q = float('nan')
        
    results_summary.append({
        "Model": "Hybrid CNN-QSVC", "OOD_Accuracy": acc_q, "OOD_F1": f1_q, "OOD_Precision": precision_score(ninja_labels_qsvc, qsvc_preds, zero_division=0),
        "OOD_Recall": recall_score(ninja_labels_qsvc, qsvc_preds, zero_division=0), "OOD_AUC": auc_q, "FPS": float('nan'), "Params": 1024
    })
    results_raw["Hybrid CNN-QSVC"] = {
        'labels': ninja_labels_qsvc, 'preds': qsvc_preds, 'probs': qsvc_probs,
        'history': {'acc': [0]*15}, 'fps': float('nan'), 'params': 1024
    }

    # 6. Save Summary and Generate Plots
    df = pd.DataFrame(results_summary)
    summary_path = "final_benchmark_metrics.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n🎉 Benchmark Suite Complete! Summary saved to {summary_path}")
    print(df.to_string())

    print("\nGenerating Plots...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    from math import pi

    plot_dir = "benchmark_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Barplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.barplot(data=df, x='Model', y='OOD_Accuracy', ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title("Model Accuracy Comparison (OOD)")
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df, x='Model', y='FPS', ax=axes[0, 1], palette='rocket')
    axes[0, 1].set_title("Inference Speed (FPS)")
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(data=df, x='Model', y='Params', ax=axes[1, 0], palette='mako')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Model Complexity (Total Parameters)")
    axes[1, 0].tick_params(axis='x', rotation=45)

    for name, res in results_raw.items():
        if name not in ["YOLOv8n-cls", "Hybrid CNN-QSVC"]:
            axes[1, 1].plot(res['history']['acc'], marker='o', label=name)
    axes[1, 1].set_title("Training Accuracy over Epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "core_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Radar Chart
    categories = ['OOD_Accuracy', 'OOD_F1', 'OOD_AUC', 'Speed (FPS)', 'Efficiency (1/Params)']
    N = len(categories)
    df_radar = df.copy()
    # Handle NaN FPS for QSVC by filling with 0 or min
    df_radar['FPS'] = df_radar['FPS'].fillna(0)
    df_radar['Speed (FPS)'] = df_radar['FPS'] / (df_radar['FPS'].max() + 1e-9)
    df_radar['Efficiency (1/Params)'] = (1/df_radar['Params']) / ((1/df_radar['Params']).max() + 1e-9)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    
    for i, row in df_radar.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Radar Comparison of All Models")
    plt.savefig(os.path.join(plot_dir, "radar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for name, res in results_raw.items():
        if len(set(res['labels'])) > 1: # check if both classes are present
            fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(res['labels'], res['probs']):.3f})")
            prec, rec, _ = precision_recall_curve(res['labels'], res['probs'])
            ax2.plot(rec, prec, label=name)
            
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Confusion Matrices
    import math
    num_models = len(results_raw)
    cols = 4
    rows = math.ceil(num_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    if num_models > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for idx, (name, res) in enumerate(results_raw.items()):
        cm = confusion_matrix(res['labels'], res['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{name}\nConfusion Matrix")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['Corn', 'Weed'])
        axes[idx].set_yticklabels(['Corn', 'Weed'])
        
    for idx in range(num_models, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ All plots successfully saved to the '{plot_dir}' directory!")

if __name__ == '__main__':
    main()
