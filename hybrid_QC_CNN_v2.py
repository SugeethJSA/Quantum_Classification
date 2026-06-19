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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, confusion_matrix
from PIL import Image
import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("aminelaatam/weed-classification")
print(f"Dataset path: {path}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), # EfficientNet performs better on slightly larger images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CornWeedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_map = {"Corn": 0, "Weed": 1, "corn": 0, "weed": 1}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_map.get(class_name.title(), -1))

        valid_indices = [i for i, label in enumerate(self.labels) if label != -1]
        self.images = [self.images[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load datasets
train_path = os.path.join(path, "CornWeed", "train")
test_path = os.path.join(path, "CornWeed", "test")

print("Loading datasets...")
train_dataset = CornWeedDataset(train_path, transform=transform)
test_dataset = CornWeedDataset(test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define Quantum Device
n_qubits = 4 # Using 4 qubits with Amplitude Embedding allows 2^4 = 16 features
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit with Amplitude Embedding
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Apply Amplitude Embedding ONCE at the start to encode 16 features into 4 qubits
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
    # StronglyEntanglingLayers applies all layers natively
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define Quantum Layer
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class HybridEfficientNetQNN(nn.Module):
    def __init__(self):
        super(HybridEfficientNetQNN, self).__init__()
        # Classical feature extractor
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Freeze efficientnet parameters for faster training of the QNN head
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Projection layer to match quantum inputs (2^4 = 16 features)
        self.fc1 = nn.Linear(1280, 16) 
        self.quantum = quantum_layer
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Output 16 features
        x = self.fc1(x) 
        
        # L2 Normalize for Amplitude Embedding
        # Adding eps to prevent NaN gradients when vector norm is close to 0
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        
        x_q = self.quantum(x_norm)
        x_out = self.fc2(x_q)
        return x_out

# Instantiate Model
model = HybridEfficientNetQNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

num_epochs = 5  # Quick train loop
losses = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training Complete!")

# -------------------------
# Evaluation & Diagnostics
# -------------------------
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

print("Evaluating model and extracting diagnostics...")
all_labels = []
all_predictions = []
all_probs = []

model.eval()

# Measure inference time
start_infer_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probabilities)
        
end_infer_time = time.time()

total_infer_time = end_infer_time - start_infer_time
avg_infer_time_ms = (total_infer_time / len(test_loader.dataset)) * 1000

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

# Calculate Metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)
try:
    auc_roc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc_roc = float('nan') # In case only one class is present in test set
mse = mean_squared_error(all_labels, all_probs)
rmse = np.sqrt(mse)
r2 = r2_score(all_labels, all_probs)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n--- Results ---")
print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f} | AUC-ROC: {auc_roc:.4f}")
print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")
print(f"Trainable Params: {param_count} | Avg Inference Time: {avg_infer_time_ms:.2f} ms/sample")

# Save summary
summary_df = pd.DataFrame([{
    "model": "HybridEfficientNetQNN",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "auc_roc": auc_roc,
    "rmse": rmse,
    "trainable_params": param_count,
    "total_inference_s": total_infer_time,
    "avg_inference_ms": avg_infer_time_ms
}])
summary_df.to_csv(os.path.join(out_dir, "evaluation_summary.csv"), index=False)

# Save per-sample
sample_df = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_predictions,
    "probability_class_1": all_probs
})
sample_df.to_csv(os.path.join(out_dir, "per_sample_predictions.csv"), index=False)

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix: Hybrid EfficientNet-QNN')
plt.colorbar()
plt.xticks([0, 1], ['Corn (0)', 'Weed (1)'])
plt.yticks([0, 1], ['Corn (0)', 'Weed (1)'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
plt.close()

# Plot Error Histogram
errors = all_probs - all_labels
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, color='purple', alpha=0.7)
plt.title('Prediction Error Distribution (Prob - True)')
plt.xlabel('Error')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "error_histogram.png"), dpi=150)
plt.close()

# Plot KDE / Scatter equivalent
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.kdeplot(all_probs[all_labels==0], label="Corn (True)", fill=True)
sns.kdeplot(all_probs[all_labels==1], label="Weed (True)", fill=True)
plt.title('Probability Distribution by Class')
plt.xlabel('Predicted Probability (Weed)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=150)
plt.close()

print(f"\n✅ All diagnostics and metrics saved to '{out_dir}/'")
