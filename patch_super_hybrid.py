import json
import codecs

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. ADD HYBRID TRANSFER LEARNING CLASS
hybrid_tl_code = """
# ==========================================
# Hybrid CNN-QNN (Transfer Learning)
# ==========================================
import torchvision.models as models

class HybridCNNQNN_Transfer(nn.Module):
    def __init__(self):
        super(HybridCNNQNN_Transfer, self).__init__()
        # Use pretrained ResNet18 backbone
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the classical backbone so it acts as an expert feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(512, 10)  # ResNet outputs 512 features, map to 10 qubits
        self.qnn = QuantumLayer_Legacy() # Reuse the 10-qubit quantum layer from legacy
        self.fc2 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # CPU/GPU bridging for Quantum layer
        x_cpu = x.cpu()
        self.qnn.to('cpu')
        x_q = self.qnn(x_cpu)
        
        x_q = x_q.reshape(x_q.shape[0], 1)
        x_q = x_q.to(x.device)
        x_out = self.fc2(x_q)
        return x_out

hybrid_tl = HybridCNNQNN_Transfer().to(device)
"""

# 2. ADD TIME TRACKING TO TRAIN FUNCTION
train_patch_old = """        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)"""

train_patch_new = """        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
    
    end_time = time.time()
    return {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'fps': fps,
        'params': params,
        'time': end_time - start_time
    }"""

# 3. ADD NEW PLOTS AND METRICS
plots_code = """
## 1.6. Training Time Comparison
plt.figure(figsize=(12, 6))
times = [res['time'] for res in results.values()]
sns.barplot(x=list(results.keys()), y=times, palette='magma')
plt.title('Total Training Time per Model (15 Epochs)')
plt.ylabel('Seconds')
plt.xticks(rotation=45)
plt.savefig("training_times.png", dpi=300, bbox_inches='tight')
plt.show()

## 1.7. Violin Plot of Prediction Confidences
plt.figure(figsize=(14, 6))
violin_data = []
for name, res in results.items():
    for prob in res['probs']:
        violin_data.append({'Model': name, 'Confidence': prob})
import pandas as pd
df_violin = pd.DataFrame(violin_data)
sns.violinplot(data=df_violin, x='Model', y='Confidence', palette='coolwarm', inner="quartile")
plt.title('Violin Plot Analysis of Prediction Confidences')
plt.xticks(rotation=45)
plt.savefig("violin_plot.png", dpi=300, bbox_inches='tight')
plt.show()

## 1.8. KDE Analysis of Probability Distributions
plt.figure(figsize=(14, 6))
for name, res in results.items():
    sns.kdeplot(res['probs'], label=name, fill=True, alpha=0.3)
plt.title('KDE Analysis of Prediction Confidences Across Models')
plt.xlabel('Predicted Probability (Weed)')
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("kde_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Inject Hybrid_TL model
        if 'legacy_qnn = Custom_Legacy_QNN().to(device)' in source and 'hybrid_tl =' not in source:
            source += "\n" + hybrid_tl_code + "\n"
            
        # 2. Patch train_model for time tracking
        if 'def train_model' in source:
            if 'import time' not in source:
                source = source.replace('def train_model(model, name, epochs):', 'import time\ndef train_model(model, name, epochs):\n    start_time = time.time()')
            if 'return {' in source and "'time':" not in source:
                # We need to replace the return statement to include time
                # The easiest way is to use regex or string replace
                old_return = "    return {\n        'labels': all_labels,\n        'preds': all_preds,\n        'probs': all_probs,\n        'fps': fps,\n        'params': params\n    }"
                new_return = "    end_time = time.time()\n    return {\n        'labels': all_labels,\n        'preds': all_preds,\n        'probs': all_probs,\n        'fps': fps,\n        'params': params,\n        'time': end_time - start_time\n    }"
                source = source.replace(old_return, new_return)
                
        # 3. Update epochs and model lineup
        if 'epochs=5' in source:
            source = source.replace('epochs=5', 'epochs=15')
        
        old_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]'
        new_loop = 'for m, name in [(hybrid_model, "Hybrid CNN-QNN (From Scratch)"), (hybrid_tl, "Hybrid CNN-QNN (Transfer Learning)"), (resnet, "ResNet18"), \n                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),\n                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]'
        source = source.replace(old_loop, new_loop)
        
        # 4. Inject plots code at the end
        if 'shap.image_plot' in source and 'plt.figure(figsize=(12, 6))' not in source:
            source += "\n" + plots_code + "\n"

        # 5. Fix subplot sizes again for 7 models -> 8 models if needed, though confusion matrix has dynamic rows/cols?
        # Actually in previous patch I changed it to 1, 7. Let's make it dynamic or 1, 8.
        if 'fig, axes = plt.subplots(1, 7' in source:
            source = source.replace('fig, axes = plt.subplots(1, 7, figsize=(35, 5))', 'fig, axes = plt.subplots(1, 7, figsize=(35, 5))') # Wait, we have 7 models now! Wait, hybrid_model + hybrid_tl + resnet + mobile + efficient + legacy_cnn + legacy_qnn = 7 models!
            # Wait, no: hybrid + hybrid_tl + 3 SOTA + 2 legacy = 7. Ah wait: YOLO is run separately! So there are 7 PyTorch models in the dictionary. YOLO is added later. Total 8 models!
            # YOLO logic is later in the script. The confusion matrix loops over results.items() which contains 8.
            # So `len(results)` is 8. `fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))`
            source = re.sub(r'fig, axes = plt.subplots\(1, \d+, figsize=\(\d+, 5\)\)', 'fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))', source)

        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Patch applied for Transfer Learning, 15 Epochs, Time Tracking, and new plots!")
