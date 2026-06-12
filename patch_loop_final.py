import json

with open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source_code = """import time

def train_model(model, name, epochs=15):
    print(f"\\nTraining {name}...")
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        acc = correct / total
        history['loss'].append(running_loss / len(train_loader))
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Acc: {acc:.4f}")
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            
    fps = measure_fps(model, test_loader, device)
    params = count_parameters(model)
    
    end_time = time.time()
    
    return {
        'name': name,
        'history': history,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'fps': fps,
        'params': params,
        'time': end_time - start_time
    }

results = {}
for m, name in [(hybrid_model, "Hybrid CNN-QNN (From Scratch)"), (hybrid_tl, "Hybrid CNN-QNN (Transfer Learning)"), (resnet, "ResNet18"), 
                (mobilenet, "MobileNetV3"), (efficientnet, "EfficientNet-B0"),
                (legacy_cnn, "Custom_Legacy_CNN"), (legacy_qnn, "Custom_Legacy_QNN")]:
    results[name] = train_model(m, name, epochs=15)
"""

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_model(' in source and 'history =' in source:
            # We found the cell, replace its source entirely
            lines = new_source_code.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            print("Successfully updated the train_model and loop cell.")

with open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Patch applied perfectly!')
