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
    print(f"🚀 Running QSVC Mixed Dataset Variance Test")
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

    for run_idx in range(1, 11):
        print(f"\n=============================================")
        print(f" QSVC Variance Testing: Run {run_idx}/10")
        print(f"=============================================")
        
        # 1. Grab 50 random images from CornWeed
        subset_size = min(50, len(train_features))
        if len(np.unique(train_labels_qsvc)) > 1:
            train_features_sub, _, train_labels_sub, _ = train_test_split(
                train_features, train_labels_qsvc, train_size=subset_size, 
                stratify=train_labels_qsvc, random_state=run_idx * 42
            )
        else:
            train_features_sub = train_features[:subset_size]
            train_labels_sub = train_labels_qsvc[:subset_size]
            
        # 2. Grab 100 random images from DatasetNinja (50 for train, 50 for test)
        ninja_pool_size = min(100, len(ninja_features))
        if len(np.unique(ninja_labels_qsvc)) > 1:
            ninja_pool_feat, _, ninja_pool_lbl, _ = train_test_split(
                ninja_features, ninja_labels_qsvc, train_size=ninja_pool_size, 
                stratify=ninja_labels_qsvc, random_state=run_idx * 123
            )
        else:
            ninja_pool_feat = ninja_features[:ninja_pool_size]
            ninja_pool_lbl = ninja_labels_qsvc[:ninja_pool_size]
            
        # 3. Split the 100 Ninja images into 50 train, 50 test
        if len(np.unique(ninja_pool_lbl)) > 1:
            ninja_train_feat, ninja_test_feat, ninja_train_lbl, ninja_test_lbl = train_test_split(
                ninja_pool_feat, ninja_pool_lbl, train_size=50, 
                stratify=ninja_pool_lbl, random_state=run_idx * 456
            )
        else:
            ninja_train_feat = ninja_pool_feat[:50]
            ninja_test_feat = ninja_pool_feat[50:]
            ninja_train_lbl = ninja_pool_lbl[:50]
            ninja_test_lbl = ninja_pool_lbl[50:]
            
        # 4. Mix CornWeed (50) and Ninja Train (50) into a single 100-image Mixed Training Set
        mixed_train_feat = np.vstack([train_features_sub, ninja_train_feat])
        mixed_train_lbl = np.concatenate([train_labels_sub, ninja_train_lbl])

        print("Computing Kernel Matrix for Mixed Training Set (100x100)...")
        K_train = compute_kernel_matrix(mixed_train_feat, mixed_train_feat, symmetric=True)
        
        print("Fitting SVC on Mixed Dataset...")
        svc = SVC(kernel="precomputed", probability=True)
        svc.fit(K_train, mixed_train_lbl)
        
        os.makedirs("saved_models_variance", exist_ok=True)
        model_path = os.path.join("saved_models_variance", f"Hybrid_CNN_QSVC_Mixed_Run{run_idx}.pkl")
        joblib.dump(svc, model_path)
        
        print("Computing Kernel Matrix for OOD Testing (50x100)...")
        K_test = compute_kernel_matrix(ninja_test_feat, mixed_train_feat)
        
        print("Evaluating QSVC on remaining 50 DatasetNinja images...")
        qsvc_preds = svc.predict(K_test)
        qsvc_probs = svc.predict_proba(K_test)[:, 1]
        
        acc_q = accuracy_score(ninja_test_lbl, qsvc_preds)
        f1_q = f1_score(ninja_test_lbl, qsvc_preds, zero_division=0)
        prec_q = precision_score(ninja_test_lbl, qsvc_preds, zero_division=0)
        rec_q = recall_score(ninja_test_lbl, qsvc_preds, zero_division=0)
        try:
            auc_q = roc_auc_score(ninja_test_lbl, qsvc_probs)
        except ValueError:
            auc_q = float('nan')
        
        print(f"\n--- QSVC FINAL METRICS (Run {run_idx}) ---")
        print(f"  --> Acc: {acc_q:.4f} | Prec: {prec_q:.4f} | Rec: {rec_q:.4f} | F1: {f1_q:.4f} | AUC: {auc_q:.4f}")

        # Append to a NEW CSV so we don't overwrite the previous one
        csv_path = "variance_mixed_benchmark_metrics.csv"
        new_row = {"Model": f"Mixed Dataset QSVC (Legit Run {run_idx})", "OOD_Accuracy": acc_q, "OOD_F1": f1_q, "OOD_Precision": prec_q, 
                   "OOD_Recall": rec_q, "OOD_AUC": auc_q, "FPS": float('nan'), "Params": 1024}
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
            
        df.to_csv(csv_path, index=False)
        print(f"Saved results for Run {run_idx} to {csv_path}")

    # 5. Generate Quantum Graphs for Hackathon
    print("\nGenerating Quantum Feature Map and Kernel Heatmap Graphs...")
    os.makedirs("benchmark_plots_variance", exist_ok=True)
    
    # 5a. Quantum Kernel Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(K_train, cmap='viridis', square=True, cbar_kws={'label': 'Quantum Kernel Value'})
    plt.title("Mixed Quantum Kernel Matrix (100x100)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join("benchmark_plots_variance", "QSVC_Mixed_Kernel_Heatmap.png"), dpi=300)
    plt.close()
    print("  -> Saved benchmark_plots_variance/QSVC_Mixed_Kernel_Heatmap.png")
    
    # 5b. Quantum Feature Map Circuit Diagram
    fig, ax = qml.draw_mpl(kernel_circuit)(mixed_train_feat[0], mixed_train_feat[1])
    fig.suptitle("Quantum Kernel Feature Map Circuit", fontsize=14)
    fig.savefig(os.path.join("benchmark_plots_variance", "QSVC_Mixed_Circuit_Diagram.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved benchmark_plots_variance/QSVC_Mixed_Circuit_Diagram.png")

if __name__ == '__main__':
    main()
