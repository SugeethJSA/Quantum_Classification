import torch
import torchvision.models as models
from ultralytics import YOLO
from Benchmark_Clean_Eval import (
    Custom_Legacy_CNN, 
    HybridCNNQNN_Upgraded, 
    Custom_Legacy_QNN_Upgraded, 
    HybridCNNQNN_Transfer_Upgraded,
    n_qubits
)

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}...")
    
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    
    models_to_test = [
        ("ResNet18", models.resnet18(weights='DEFAULT')),
        ("MobileNetV3", models.mobilenet_v3_small(weights='DEFAULT')),
        ("EfficientNet-B0", models.efficientnet_b0(weights='DEFAULT')),
        ("Custom_Legacy_CNN", Custom_Legacy_CNN()),
        ("HybridCNNQNN_Upgraded", HybridCNNQNN_Upgraded()),
        ("Custom_Legacy_QNN_Upgraded", Custom_Legacy_QNN_Upgraded()),
        ("HybridCNNQNN_Transfer_Upgraded", HybridCNNQNN_Transfer_Upgraded())
    ]
    
    # Adjust classical models
    models_to_test[0][1].fc = torch.nn.Linear(models_to_test[0][1].fc.in_features, 2)
    models_to_test[1][1].classifier[3] = torch.nn.Linear(models_to_test[1][1].classifier[3].in_features, 2)
    models_to_test[2][1].classifier[1] = torch.nn.Linear(models_to_test[2][1].classifier[1].in_features, 2)
    
    for name, model in models_to_test:
        try:
            model = model.to(device)
            out = model(dummy_input)
            assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
            print(f"[OK] {name} forward pass successful. Output shape: {out.shape}")
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            
    try:
        yolo = YOLO('yolov8n-cls.pt')
        print("[OK] YOLOv8n initialized successfully.")
    except Exception as e:
        print(f"[ERROR] YOLO failed: {e}")

if __name__ == "__main__":
    test_models()
