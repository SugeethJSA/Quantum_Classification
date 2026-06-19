import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import shap
import warnings
warnings.filterwarnings('ignore')

from Benchmark_Clean_Eval import (
    DatasetNinjaClassification,
    HybridCNNQNN_Transfer_Upgraded,
    to_wsl_path,
    n_qubits,
    create_upgraded_quantum_layer
)

# 1. Hardcoded Training Data from the User's logs
training_data = {
    "ResNet18": [0.9965, 0.9895, 0.9955, 0.9930, 0.9920, 0.9925, 0.9965, 0.9965, 0.9995, 1.0000, 1.0000, 0.9980, 0.9970, 0.9990, 0.9925],
    "MobileNetV3": [0.9905, 0.9955, 0.9975, 1.0000, 0.9955, 1.0000, 0.9995, 0.9995, 0.9915, 0.9980, 1.0000, 1.0000, 0.9995, 0.9995, 0.9990],
    "EfficientNet-B0": [0.9960, 0.9980, 0.9995, 1.0000, 1.0000, 0.9995, 0.9995, 0.9995, 1.0000, 1.0000, 0.9995, 1.0000, 1.0000, 0.9995, 0.9975],
    "Custom_Legacy_CNN": [0.9057, 0.9393, 0.9328, 0.9669, 0.9724, 0.9769, 0.9809, 0.9814, 0.9849, 0.9890, 0.9900, 0.9865, 0.9915, 0.9940, 0.9920],
    "Hybrid CNN-QNN (From Scratch)": [0.9644, 0.9528, 0.9759, 0.9378, 0.9784, 0.9799, 0.9704, 0.9915, 0.9920, 0.9905, 0.9794, 0.9900, 0.9839, 0.9754, 0.8695],
    "Custom_Legacy_QNN": [0.7311, 0.8746, 0.9363, 0.9142, 0.9147, 0.9207, 0.9237, 0.9373, 0.9443, 0.9272, 0.9313, 0.9222, 0.9267, 0.9438, 0.9488],
    "Hybrid CNN-QNN (Transfer Learning)": [0.9644, 0.9739, 0.9769, 0.9799, 0.9860, 0.9829, 0.9860, 0.9895, 0.9839, 0.9875, 0.9880, 0.9870, 0.9895, 0.9945, 0.9910]
}

def plot_training_curves():
    os.makedirs('benchmark_sota', exist_ok=True)
    plt.figure(figsize=(12, 7))
    epochs = range(1, 16)
    
    for model_name, acc_list in training_data.items():
        linewidth = 3 if "QNN" in model_name else 1.5
        linestyle = '-' if "QNN" in model_name else '--'
        plt.plot(epochs, acc_list, label=model_name, linewidth=linewidth, linestyle=linestyle, marker='o', markersize=4)
        
    plt.title('Validation Accuracy over Epochs (Classical vs Hybrid Quantum)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0.7, 1.01)
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('benchmark_sota', 'Training_Curve_ValAcc.png'), dpi=300)
    plt.close()
    print("[SUCCESS] Generated Training_Curve_ValAcc.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Plot Training Curves first
    plot_training_curves()
    
    # 2. Setup Data Loader for Inference
    ninja_ann_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\ann")
    ninja_img_dir = to_wsl_path(r"C:\Users\inaug\Downloads\maize-weed-image-DatasetNinja\ds\img")
    
    # Same normalizations used in Benchmark_Clean_Eval
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load just a subset of OOD dataset (e.g., first 500 images) to speed up Inference for SHAP/ROC
    ninja_dataset = DatasetNinjaClassification(ninja_ann_dir, ninja_img_dir, transform=test_transform, is_test=True, split_dir=r"C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit")
    # Subsample to 500 images for fast PR/ROC calculation
    subset_indices = np.random.choice(len(ninja_dataset), min(500, len(ninja_dataset)), replace=False)
    sub_dataset = torch.utils.data.Subset(ninja_dataset, subset_indices)
    ninja_loader = DataLoader(sub_dataset, batch_size=32, shuffle=False)

    print(f"Loaded {len(sub_dataset)} OOD test images for Inference.")

    # We will instantiate two top models for ROC/PR/SHAP: EfficientNet-B0 and Hybrid CNN-QNN (Transfer Learning)
    # We choose these to represent the best of Classical vs Best of Quantum
    models_dict = {}
    
    # Load EfficientNet
    effnet = models.efficientnet_b0(weights=None)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 2)
    effnet.load_state_dict(torch.load(os.path.join("saved_models", "EfficientNet-B0.pth"), map_location=device))
    effnet.to(device)
    effnet.eval()
    models_dict['EfficientNet-B0 (Classical SOTA)'] = effnet

    # Load Hybrid CNN-QNN
    hybrid_qnn = HybridCNNQNN_Transfer_Upgraded()
    hybrid_qnn.load_state_dict(torch.load(os.path.join("saved_models", "Hybrid_CNN-QNN_(Transfer_Learning).pth"), map_location=device))
    hybrid_qnn.to(device)
    hybrid_qnn.eval()
    models_dict['Hybrid CNN-QNN (Quantum Transfer)'] = hybrid_qnn

    y_true_all = []
    y_probs_dict = {name: [] for name in models_dict.keys()}

    print("Running Inference over Test Set...")
    with torch.no_grad():
        for inputs, labels in ninja_loader:
            inputs = inputs.to(device)
            labels_np = labels.numpy()
            y_true_all.extend(labels_np)
            
            for name, model in models_dict.items():
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_probs_dict[name].extend(probs)
                
    y_true_all = np.array(y_true_all)
    
    # --- ROC CURVES ---
    plt.figure(figsize=(10, 8))
    for name, probs in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true_all, probs)
        roc_auc = auc(fpr, tpr)
        color = '#9b59b6' if 'Quantum' in name else '#3498db'
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2.5)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('benchmark_sota', 'ROC_Curve_Comparison.png'), dpi=300)
    plt.close()
    print("[SUCCESS] Generated ROC_Curve_Comparison.png")

    # --- PR CURVES ---
    plt.figure(figsize=(10, 8))
    for name, probs in y_probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true_all, probs)
        color = '#9b59b6' if 'Quantum' in name else '#3498db'
        plt.plot(recall, precision, label=name, color=color, linewidth=2.5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('benchmark_sota', 'PR_Curve_Comparison.png'), dpi=300)
    plt.close()
    print("[SUCCESS] Generated PR_Curve_Comparison.png")

    # --- MODEL AGREEMENT (Correlation Heatmap) ---
    df_probs = pd.DataFrame(y_probs_dict)
    corr_matrix = df_probs.corr(method='spearman')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0.5, vmax=1.0, fmt=".3f", square=True)
    plt.title('Model Agreement (Spearman Rank Correlation)')
    plt.tight_layout()
    plt.savefig(os.path.join('benchmark_sota', 'Model_Agreement_Heatmap.png'), dpi=300)
    plt.close()
    print("[SUCCESS] Generated Model_Agreement_Heatmap.png")

    # --- SHAP VALUES (Feature Importance) ---
    print("\nInitializing SHAP Explainer (This may take a moment)...")
    
    # Grab background images and test images
    # We take 10 images for the background, 2 images for the test
    bg_inputs, _ = next(iter(DataLoader(sub_dataset, batch_size=10, shuffle=True)))
    bg_inputs = bg_inputs.to(device)
    
    test_inputs, _ = next(iter(DataLoader(sub_dataset, batch_size=2, shuffle=True)))
    test_inputs = test_inputs.to(device)
    
    # Denormalize function to plot the raw image back
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean
        
    test_images_np = denormalize(test_inputs).cpu().numpy().transpose(0, 2, 3, 1)

    # 1. SHAP for EfficientNet-B0
    try:
        effnet_explainer = shap.GradientExplainer(effnet, bg_inputs)
        shap_values_effnet = effnet_explainer.shap_values(test_inputs)
        
        def format_shap(sv):
            if isinstance(sv, list):
                return [np.transpose(s, (0, 2, 3, 1)) if s.shape[1] == 3 else s for s in sv]
            else:
                return np.transpose(sv, (0, 2, 3, 1)) if sv.shape[1] == 3 else sv

        shap_values_effnet = format_shap(shap_values_effnet)
            
        # Plot and save
        shap.image_plot(shap_values_effnet, test_images_np, show=False)
        plt.savefig(os.path.join('benchmark_sota', 'SHAP_EfficientNet.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("[SUCCESS] Generated SHAP_EfficientNet.png")
    except Exception as e:
        print(f"[WARNING] SHAP for EfficientNet failed: {e}")

    # 2. SHAP for Hybrid CNN-QNN
    # Note: GradientExplainer sometimes struggles tracing back through complex quantum autodiff layers
    try:
        qnn_explainer = shap.GradientExplainer(hybrid_qnn, bg_inputs)
        shap_values_qnn = qnn_explainer.shap_values(test_inputs)
        
        shap_values_qnn = format_shap(shap_values_qnn)
            
        shap.image_plot(shap_values_qnn, test_images_np, show=False)
        plt.savefig(os.path.join('benchmark_sota', 'SHAP_HybridQNN.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("[SUCCESS] Generated SHAP_HybridQNN.png")
    except Exception as e:
        print(f"[WARNING] SHAP for Hybrid CNN-QNN failed due to Quantum Node gradients: {e}")
        # Alternative approach for Hybrid models if gradients don't flow cleanly: Partial SHAP
        # Or we skip the quantum SHAP plot if PennyLane doesn't support Torch backprop in this specific Explainer mode.

if __name__ == '__main__':
    main()
