import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

from Benchmark_Clean_Eval import (
    to_wsl_path,
    DatasetNinjaClassification,
    CornWeedDataset,
    HybridCNNQNN_Transfer_Upgraded,
    n_qubits
)
import kagglehub
import torchvision.transforms as transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=============================================")
    print(f"🚀 Running QSVC Benchmark Standalone")
    print(f"=============================================")
    
    # 0. Setup Dataset Paths and Transforms
    path = kagglehub.dataset_download("aminelaatam/weed-classification")
    DATA_DIR = os.path.join(path, "CornWeed")
    SPLIT_DIR = os.path.join(os.path.dirname(DATA_DIR), "CornWeed_CleanSplit")

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
    
    # 1. Load Data
    print("Loading datasets...")
    ninja_ann_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\ann")
    ninja_img_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\img")
    
    ninja_dataset = DatasetNinjaClassification(ninja_ann_dir, ninja_img_dir, transform=test_transform, is_test=True, split_dir=SPLIT_DIR)
    ninja_loader = DataLoader(ninja_dataset, batch_size=32, shuffle=False)
    
    train_dataset = CornWeedDataset(os.path.join(SPLIT_DIR, "train"), transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Load Model
    print("Loading pretrained Hybrid CNN-QNN feature extractor...")
    model = HybridCNNQNN_Transfer_Upgraded().to(device)
    state_dict = torch.load(os.path.join("saved_models", "Hybrid_CNN-QNN_(Transfer_Learning).pth"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    def extract_features(dataloader, model):
        all_features, all_lbls = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                x = model.features(images.to(device))
                x = model.avgpool(x).view(x.size(0), -1)
                x = torch.nn.functional.normalize(model.fc1(x), p=2, dim=1, eps=1e-8)
                all_features.append(x.cpu().numpy())
                all_lbls.append(labels.numpy())
        return np.vstack(all_features), np.concatenate(all_lbls)

    print("Extracting features (this may take a minute)...")
    train_features, train_labels_qsvc = extract_features(train_loader, model)
    ninja_features, ninja_labels_qsvc = extract_features(ninja_loader, model)

    subset_size = min(50, len(train_features))
    if len(np.unique(train_labels_qsvc)) > 1:
        train_features_sub, _, train_labels_sub, _ = train_test_split(
            train_features, train_labels_qsvc, train_size=subset_size, 
            stratify=train_labels_qsvc, random_state=42
        )
    else:
        train_features_sub = train_features[:subset_size]
        train_labels_sub = train_labels_qsvc[:subset_size]
        
    # Limit ninja_features (OOD) to 50 as well so the test matrix isn't 400x50
    ninja_subset_size = min(50, len(ninja_features))
    ninja_features_sub = ninja_features[:ninja_subset_size]
    ninja_labels_sub = ninja_labels_qsvc[:ninja_subset_size]

    # 3. QSVC Kernel Calculation
    # We switch to 'lightning.qubit' which is a highly optimized C++ state-vector simulator
    dev_kernel = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev_kernel, interface="autograd")
    def kernel_circuit(x1, x2):
        qml.AmplitudeEmbedding(features=x1, wires=range(n_qubits), normalize=True, pad_with=0.)
        qml.adjoint(qml.AmplitudeEmbedding)(features=x2, wires=range(n_qubits), normalize=True, pad_with=0.)
        return qml.probs(wires=range(n_qubits))

    def compute_kernel_matrix(A, B, symmetric=False):
        print(f"  -> Computing {len(A)}x{len(B)} kernel matrix using TRUE Quantum Simulation...")
        
        # Unleash the i9 Processor: Distribute the independent quantum circuits across all CPU cores
        from joblib import Parallel, delayed
        
        def compute_row(i):
            row_data = np.zeros(len(B))
            for j in range(len(B)):
                if symmetric and j < i:
                    row_data[j] = np.nan # Placeholder, will mirror later
                elif symmetric and j == i:
                    row_data[j] = 1.0
                else:
                    # lightning.qubit returns a float64 directly, so we don't call .numpy()
                    row_data[j] = float(kernel_circuit(A[i], B[j])[0])
            return i, row_data

        # n_jobs=-1 tells joblib to use 100% of available CPU cores
        results = Parallel(n_jobs=-1, verbose=5)(delayed(compute_row)(i) for i in range(len(A)))
        
        matrix = np.zeros((len(A), len(B)))
        for i, row in results:
            matrix[i, :] = row
            
        # If symmetric, mirror the nan placeholders
        if symmetric:
            for i in range(len(A)):
                for j in range(i):
                    matrix[i, j] = matrix[j, i]
                    
        return matrix

    print("Computing Kernel Matrix for Training...")
    K_train = compute_kernel_matrix(train_features_sub, train_features_sub)
    
    print("Fitting SVC...")
    svc = SVC(kernel="precomputed", probability=True)
    svc.fit(K_train, train_labels_sub)
    
    model_path = os.path.join("saved_models", "Hybrid_CNN_QSVC.pkl")
    joblib.dump(svc, model_path)
    print(f"✅ QSVC Model saved to {model_path}")
    
    print("Computing Kernel Matrix for OOD Testing...")
    K_test = compute_kernel_matrix(ninja_features_sub, train_features_sub)
    
    print("Evaluating QSVC on OOD DatasetNinja...")
    qsvc_preds = svc.predict(K_test)
    qsvc_probs = svc.predict_proba(K_test)[:, 1]
    
    acc_q = accuracy_score(ninja_labels_sub, qsvc_preds)
    f1_q = f1_score(ninja_labels_sub, qsvc_preds, zero_division=0)
    prec_q = precision_score(ninja_labels_sub, qsvc_preds, zero_division=0)
    rec_q = recall_score(ninja_labels_sub, qsvc_preds, zero_division=0)
    try:
        auc_q = roc_auc_score(ninja_labels_sub, qsvc_probs)
    except ValueError:
        auc_q = float('nan')
    
    print(f"\n--- QSVC FINAL METRICS ---")
    print(f"  --> Acc: {acc_q:.4f} | Prec: {prec_q:.4f} | Rec: {rec_q:.4f} | F1: {f1_q:.4f} | AUC: {auc_q:.4f}")

    # Append to existing CSV if it exists
    csv_path = "final_benchmark_metrics.csv"
    new_row = {"Model": "Hybrid CNN-QSVC (Legit Simulation)", "OOD_Accuracy": acc_q, "OOD_F1": f1_q, "OOD_Precision": prec_q, 
               "OOD_Recall": rec_q, "OOD_AUC": auc_q, "FPS": float('nan'), "Params": 1024}
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
        
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # 5. Generate Quantum Graphs for Hackathon
    print("\nGenerating Quantum Feature Map and Kernel Heatmap Graphs...")
    os.makedirs("benchmark_plots", exist_ok=True)
    
    # 5a. Quantum Kernel Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(K_train, cmap='viridis', square=True, cbar_kws={'label': 'Quantum Kernel Value'})
    plt.title("Quantum Kernel Matrix (Training Subset)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join("benchmark_plots", "QSVC_Kernel_Heatmap.png"), dpi=300)
    plt.close()
    print("  -> Saved benchmark_plots/QSVC_Kernel_Heatmap.png")
    
    # 5b. Quantum Feature Map Circuit Diagram
    # We draw the circuit using the first two samples from the dataset
    fig, ax = qml.draw_mpl(kernel_circuit)(train_features_sub[0], train_features_sub[1])
    fig.suptitle("Quantum Kernel Feature Map Circuit", fontsize=14)
    fig.savefig(os.path.join("benchmark_plots", "QSVC_Circuit_Diagram.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved benchmark_plots/QSVC_Circuit_Diagram.png")

if __name__ == '__main__':
    main()
