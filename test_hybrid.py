import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (2, n_qubits, 3)})

class HybridCNNQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 32 * 32, n_qubits)
        self.quantum = quantum_layer
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        
        x_cpu = x.cpu()
        self.quantum.to("cpu")
        x_q = self.quantum(x_cpu)
        
        x_q = x_q.to(x.device)
        return self.fc2(x_q)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = HybridCNNQNN().to(device)

dummy_input = torch.randn(2, 3, 128, 128).to(device)
try:
    out = model(dummy_input)
    print("Forward pass successful!", out.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
