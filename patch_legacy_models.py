import json
import codecs

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

legacy_code = """
# ==========================================
# Legacy Custom CNN from Main.ipynb
# ==========================================
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
        x = self.fc2(x)
        return x

legacy_cnn = Custom_Legacy_CNN().to(device)

# ==========================================
# Legacy QNN from Main.ipynb
# ==========================================
n_qubits_legacy = 10
dev_legacy = qml.device("default.qubit", wires=n_qubits_legacy)

@qml.qnode(dev_legacy, interface="torch")
def quantum_circuit_legacy(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits_legacy))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits_legacy))
    return qml.expval(qml.PauliZ(0))

class QuantumLayer_Legacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit_legacy, {"weights": (2, n_qubits_legacy, 3)})

    def forward(self, x):
        return self.q_layer(x)

class Custom_Legacy_QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, n_qubits_legacy)
        self.qnn = QuantumLayer_Legacy()
        self.fc2 = nn.Linear(1, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # FIX FOR CPU-GPU MISMATCH
        x_cpu = x.cpu()
        self.qnn.to('cpu')
        x_q = self.qnn(x_cpu)
        
        x_q = x_q.reshape(x_q.shape[0], 1)
        x_q = x_q.to(x.device)
        x_out = self.fc2(x_q)
        return x_out  # REMOVED SOFTMAX FOR CROSS ENTROPY LOSS!

legacy_qnn = Custom_Legacy_QNN().to(device)
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Insert the legacy models right after the Hybrid model instantiation
        if 'class HybridCNNQNN(nn.Module):' in source and 'legacy_cnn =' not in source:
            source += "\n" + legacy_code + "\n"
            
        # Update the training loop to include the legacy models
        if 'for m, name in [(hybrid_model, "Hybrid CNN-QNN")' in source:
            old_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0")]:'
            new_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]:'
            source = source.replace(old_loop, new_loop)
            
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# Adjust subplots size for confusion matrices since we now have 7 models instead of 5
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'fig, axes = plt.subplots(1, 5, figsize=(25, 5))' in source:
            source = source.replace('fig, axes = plt.subplots(1, 5, figsize=(25, 5))', 'fig, axes = plt.subplots(1, 7, figsize=(35, 5))')
            lines = source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Legacy models successfully injected!")
