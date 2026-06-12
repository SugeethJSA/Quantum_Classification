import json
import re

with open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create new cell for Hybrid Transfer
hybrid_tl_cell = {
  'cell_type': 'code',
  'execution_count': None,
  'metadata': {},
  'outputs': [],
  'source': [
    '# ==========================================\n',
    '# Hybrid CNN-QNN (Transfer Learning)\n',
    '# ==========================================\n',
    'import torchvision.models as models\n',
    '\n',
    'class HybridCNNQNN_Transfer(nn.Module):\n',
    '    def __init__(self):\n',
    '        super(HybridCNNQNN_Transfer, self).__init__()\n',
    '        resnet = models.resnet18(pretrained=True)\n',
    '        self.features = nn.Sequential(*list(resnet.children())[:-1])\n',
    '        \n',
    '        for param in self.features.parameters():\n',
    '            param.requires_grad = False\n',
    '            \n',
    '        self.fc1 = nn.Linear(512, 10)\n',
    '        self.qnn = QuantumLayer_Legacy()\n',
    '        self.fc2 = nn.Linear(1, 2)\n',
    '\n',
    '    def forward(self, x):\n',
    '        x = self.features(x)\n',
    '        x = x.view(x.size(0), -1)\n',
    '        x = self.fc1(x)\n',
    '        \n',
    '        x_cpu = x.cpu()\n',
    '        self.qnn.to(\'cpu\')\n',
    '        x_q = self.qnn(x_cpu)\n',
    '        \n',
    '        x_q = x_q.reshape(x_q.shape[0], 1)\n',
    '        x_q = x_q.to(x.device)\n',
    '        x_out = self.fc2(x_q)\n',
    '        return x_out\n',
    '\n',
    'hybrid_tl = HybridCNNQNN_Transfer().to(device)\n'
  ]
}

# Create new cell for KDE and Violin plots
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

# Track what to modify
has_hybrid_tl = False
has_plots = False

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'HybridCNNQNN_Transfer' in src:
            has_hybrid_tl = True
        if 'KDE Analysis' in src:
            has_plots = True
        
        # Modify train_model 
        if 'def train_model(model, name, epochs):' in src:
            if 'import time' not in src:
                src = src.replace('def train_model(model, name, epochs):', 'import time\ndef train_model(model, name, epochs):\n    start_time = time.time()')
            if 'fps = sum(fps_list) / len(fps_list)' in src:
                src = src.replace('        fps = sum(fps_list) / len(fps_list)', '        fps = sum(fps_list) / len(fps_list)\n    end_time = time.time()')
            if 'return {' in src and "'time':" not in src:
                src = src.replace("'params': params", "'params': params,\n        'time': end_time - start_time")
            
            # fix epochs
            if 'epochs=5' in src:
                src = src.replace('epochs=5', 'epochs=15')
                
            # fix loop lineup
            if 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"),' in src:
                old_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]:'
                new_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN (From Scratch)"), (hybrid_tl, "Hybrid CNN-QNN (Transfer Learning)"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]:'
                src = src.replace(old_loop, new_loop)
            
            # Subplot size fix for confusion matrices
            if 'fig, axes = plt.subplots(1, 7' in src:
                src = src.replace('fig, axes = plt.subplots(1, 7, figsize=(35, 5))', 'fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))')
                
            lines = src.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

if not has_hybrid_tl:
    # Insert hybrid_tl_cell after the legacy QNN code
    idx_to_insert = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'legacy_qnn =' in ''.join(cell['source']):
            idx_to_insert = i + 1
            break
    if idx_to_insert != -1:
        nb['cells'].insert(idx_to_insert, hybrid_tl_cell)
        print('Inserted Hybrid Transfer Learning Cell')

if not has_plots:
    # Append plots at end
    nb['cells'].append(plots_cell)
    print('Inserted Plots Cell')

with open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Patching complete!')
