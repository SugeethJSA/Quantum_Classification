import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, mean_squared_error, r2_score, roc_curve
from PIL import Image
import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("aminelaatam/weed-classification")
print(f"Dataset path: {path}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Image Transformations
# FIXED: 3 channels for RGB images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    # Convert the first letter to uppercase to match the keys
                    self.labels.append(self.class_map.get(class_name.title(), -1))

        # Filter out unknown classes (-1)
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
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define Quantum Layer
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class HybridCNNQNN(nn.Module):
    def __init__(self):
        super(HybridCNNQNN, self).__init__()
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 2)  # Reduce features to match quantum layer input
        self.quantum = quantum_layer
        self.fc2 = nn.Linear(2, 2)  # Output layer (Corn vs. Weed)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.tanh(self.fc1(x))  # Match quantum input range
        x = self.quantum(x)  # Quantum processing
        x = self.fc2(x)  # Classical post-processing
        # Notice: No softmax here, because we use CrossEntropyLoss for training.
        return x

# Instantiate Model
model = HybridCNNQNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Reduced to 10 for a quicker test run. Set to 20 for full training.
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

# Evaluate
print("Evaluating model...")
all_labels = []
all_predictions = []
all_probs = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # Apply softmax for evaluation probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probabilities)

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

# Calculate Metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
auc_roc = roc_auc_score(all_labels, all_probs)
mse = mean_squared_error(all_labels, all_probs)
rmse = np.sqrt(mse)
r2 = r2_score(all_labels, all_probs)

def quantum_fidelity(p, q):
    """Correct Bhattacharyya-style fidelity for probability distributions."""
    p_norm = p / np.sum(p) if np.sum(p) > 0 else p
    q_norm = q / np.sum(q) if np.sum(q) > 0 else q
    return (np.sum(np.sqrt(p_norm * q_norm))) ** 2

q_fidelity = quantum_fidelity(all_labels, all_probs)

print(f"\n--- Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Quantum Fidelity (Bhattacharyya): {q_fidelity:.4f}")
