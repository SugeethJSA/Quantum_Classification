import json
import re

with open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

legacy_and_tl_code = """
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
        
        x_cpu = x.cpu()
        self.qnn.to('cpu')
        x_q = self.qnn(x_cpu)
        
        x_q = x_q.reshape(x_q.shape[0], 1)
        x_q = x_q.to(x.device)
        x_out = self.fc2(x_q)
        return x_out

legacy_qnn = Custom_Legacy_QNN().to(device)

# ==========================================
# Hybrid CNN-QNN (Transfer Learning)
# ==========================================
import torchvision.models as models

class HybridCNNQNN_Transfer(nn.Module):
    def __init__(self):
        super(HybridCNNQNN_Transfer, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(512, 10)
        self.qnn = QuantumLayer_Legacy()
        self.fc2 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        x_cpu = x.cpu()
        self.qnn.to('cpu')
        x_q = self.qnn(x_cpu)
        
        x_q = x_q.reshape(x_q.shape[0], 1)
        x_q = x_q.to(x.device)
        x_out = self.fc2(x_q)
        return x_out

hybrid_tl = HybridCNNQNN_Transfer().to(device)
"""

plots_cell = {
  'cell_type': 'code',
  'execution_count': None,
  'metadata': {},
  'outputs': [],
  'source': [
    '## 1.6. Training Time Comparison\n',
    'plt.figure(figsize=(12, 6))\n',
    'times = [res.get(\'time\', 0) for res in results.values()]\n',
    'sns.barplot(x=list(results.keys()), y=times, palette=\'magma\')\n',
    'plt.title(\'Total Training Time per Model (15 Epochs)\')\n',
    'plt.ylabel(\'Seconds\')\n',
    'plt.xticks(rotation=45)\n',
    'plt.savefig("training_times.png", dpi=300, bbox_inches=\'tight\')\n',
    'plt.show()\n',
    '\n',
    '## 1.7. Violin Plot of Prediction Confidences\n',
    'plt.figure(figsize=(14, 6))\n',
    'violin_data = []\n',
    'for name, res in results.items():\n',
    '    if \'probs\' in res:\n',
    '        for prob in res[\'probs\']:\n',
    '            violin_data.append({\'Model\': name, \'Confidence\': prob})\n',
    'import pandas as pd\n',
    'df_violin = pd.DataFrame(violin_data)\n',
    'sns.violinplot(data=df_violin, x=\'Model\', y=\'Confidence\', palette=\'coolwarm\', inner="quartile")\n',
    'plt.title(\'Violin Plot Analysis of Prediction Confidences\')\n',
    'plt.xticks(rotation=45)\n',
    'plt.savefig("violin_plot.png", dpi=300, bbox_inches=\'tight\')\n',
    'plt.show()\n',
    '\n',
    '## 1.8. KDE Analysis of Probability Distributions\n',
    'plt.figure(figsize=(14, 6))\n',
    'for name, res in results.items():\n',
    '    if \'probs\' in res:\n',
    '        sns.kdeplot(res[\'probs\'], label=name, fill=True, alpha=0.3)\n',
    'plt.title(\'KDE Analysis of Prediction Confidences Across Models\')\n',
    'plt.xlabel(\'Predicted Probability (Weed)\')\n',
    'plt.ylabel(\'Density\')\n',
    'plt.legend(bbox_to_anchor=(1.05, 1), loc=\'upper left\')\n',
    'plt.savefig("kde_analysis.png", dpi=300, bbox_inches=\'tight\')\n',
    'plt.show()\n'
  ]
}

has_plots = False

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        
        if 'KDE Analysis' in src:
            has_plots = True
            
        # 1. Insert Legacy + TL models
        if 'class HybridCNNQNN(nn.Module):' in src and 'Custom_Legacy_CNN' not in src:
            src += "\n" + legacy_and_tl_code + "\n"
            
        # 2. Modify train_model logic
        if 'def train_model(model, name, epochs):' in src:
            if 'import time' not in src:
                src = src.replace('def train_model(model, name, epochs):', 'import time\ndef train_model(model, name, epochs):\n    start_time = time.time()')
            if 'fps = sum(fps_list) / len(fps_list)' in src and 'end_time =' not in src:
                src = src.replace('        fps = sum(fps_list) / len(fps_list)', '        fps = sum(fps_list) / len(fps_list)\n    end_time = time.time()')
            if 'return {' in src and "'time':" not in src:
                src = src.replace("'params': params", "'params': params,\n        'time': end_time - start_time")
            
            # fix epochs
            if 'epochs=5' in src:
                src = src.replace('epochs=5', 'epochs=15')
                
            # fix loop lineup
            if 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"),' in src or 'for m, name in [(hybrid_model, "Hybrid CNN-QNN (From Scratch)"),' in src:
                # Replace whatever loop is there with the definitive 8-model array
                src = re.sub(r'for m, name in \[\(hybrid_model,.*\]:', 'for m, name in [(hybrid_model, "Hybrid CNN-QNN (From Scratch)"), (hybrid_tl, "Hybrid CNN-QNN (Transfer Learning)"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]:', src, flags=re.DOTALL)
            
            # Subplot size fix for confusion matrices
            if 'fig, axes = plt.subplots(1, 5' in src:
                src = src.replace('fig, axes = plt.subplots(1, 5, figsize=(25, 5))', 'fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))')
            if 'fig, axes = plt.subplots(1, 7' in src:
                src = src.replace('fig, axes = plt.subplots(1, 7, figsize=(35, 5))', 'fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))')
                
        lines = src.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

if not has_plots:
    nb['cells'].append(plots_cell)
    print('Inserted Plots Cell')

with open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Patching everything complete!')
