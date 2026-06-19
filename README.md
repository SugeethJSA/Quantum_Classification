# 🚀 Quantum-Enhanced Computer Vision for Weed Classification (IBM QFF25)

Welcome to the definitive repository for the **Hybrid Classical-Quantum Computer Vision** architecture, designed for extreme Out-Of-Distribution (OOD) generalization in agricultural datasets (Corn vs. Weed).

This project was built for the **IBM Quantum Hackathon** and systematically benchmarks State-Of-The-Art (SOTA) classical architectures against a novel **Hybrid CNN-Transfer-Quantum QSVC** pipeline. 

## 🧠 The Architecture: A Defense of the Hybrid QSVC Approach

Why introduce Quantum Computing into Computer Vision? Standard deep learning models (like ResNet, MobileNet, EfficientNet, and YOLOv8) achieve near-perfect training accuracies on specific datasets. However, when deployed in the real world—where lighting, soil color, and crop varietals change—they suffer catastrophic failure due to their hyper-reliance on narrow visual distributions.

Our architecture solves this by marrying three distinct paradigms:

1. **Classical Transfer Learning (EfficientNet-B0 Backbone)**
   * **Why?** Quantum computers currently lack the qubit volume to directly ingest 128x128 RGB images ($128 \times 128 \times 3 \approx 49,000$ inputs). We use the convolutional layers of EfficientNet to geometrically extract high-level feature maps (edges, textures, shapes), heavily compressing the image while retaining spatial intelligence.

2. **Quantum Feature Map (QNN & Entanglement)**
   * **Why?** Once the classical features are extracted, we use `AmplitudeEmbedding` and `StronglyEntanglingLayers` to map these features into a high-dimensional Hilbert space. The quantum nodes act as an immensely complex non-linear regularizer. By entangling the feature states, the network spreads the representations across multiple qubits, making the decision boundaries significantly more robust to classical noise (like shadows or different soil types).

3. **Quantum Support Vector Classifier (QSVC)**
   * **Why?** Instead of using a standard neural network layer to make the final prediction, we use the Quantum Circuit to calculate a **Quantum Kernel Matrix** representing the "distance" between different feature states in the Hilbert space. We then pass this kernel to a Support Vector Classifier (SVC). 
   * **The Quantum Advantage**: Unlike gradient descent, which gets stuck in local minima and perfectly memorizes the training data (causing overfitting), SVC mathematics are strictly convex. The QSVC finds the *Global Maximum Margin* hyperplane. This ensures that the quantum-separated features are divided with the absolute maximum possible generalization distance.

**The Result?** A model that takes the visual intelligence of SOTA classical CNNs, regularizes it through Quantum Entanglement, and mathematically guarantees the most robust decision boundary possible using Quantum Kernel Support Vector Mathematics. 

---

## 📊 Results & Visualization (The `benchmark_sota/` suite)

We ran an aggressive 10-Run Variance Test over independent randomized subsets of the datasets to prevent cherry-picking. The model was evaluated on a strictly unseen OOD dataset (`DatasetNinja`) after training on `CornWeedDataset`.

All analytical outputs are stored in the `benchmark_sota/` directory:

* **`ROC_Curve_Comparison.png`**: Proves our Hybrid CNN-QSVC achieves competitive Area Under the Curve (AUC), massively outperforming legacy setups.
* **`PR_Curve_Comparison.png`**: Shows precision/recall thresholds across all tested architectures.
* **`Model_Agreement_Heatmap.png`**: A Spearman Rank Correlation Heatmap. This is critical: it proves that the Quantum Models are making *different statistical decisions* than the Classical models, confirming that the quantum Hilbert mapping fundamentally altered the feature interpretation!
* **`Training_Curve_ValAcc.png`**: Tracks the Validation Accuracy across 15 epochs for all architectures.
* **`RadarChart_Highlights.png`**: A beautiful spider chart directly overlaying the Best Classical SOTA against the Quantum architecture.
* **`SHAP_EfficientNet.png`**: Pixel-level feature importance maps showing exactly which leaves the model focuses on.
* **`MASTER_BENCHMARK_RESULTS.csv`**: The consolidated metrics across all runs and models.

---

## 💻 Codebase & How to Run

### 1. `Benchmark_Clean_Eval.py` (The Classical & Hybrid QNN Baseline)
This script is the core baseline trainer. It automatically downloads the Kaggle datasets, cleans the data split (using MD5 hashing), and trains 8 different architectures (ResNet, EfficientNet, MobileNet, YOLOv8n, Custom CNN, Custom QNN, Hybrid scratch, Hybrid Transfer) over 15 Epochs using PyTorch. 
*Note: This strictly trains the neural networks, not the QSVC.*
```bash
python Benchmark_Clean_Eval.py
```

### 2. `Run_QSVC_Legit.py` (The True Quantum Kernel Pipeline)
This script loads the pre-trained `Hybrid_CNN-QNN_(Transfer_Learning).pth` model, extracts the latent features, and feeds them into the PennyLane `lightning.qubit` C++ State-Vector simulator. It aggressively parallelizes across all available CPU cores using `joblib` to calculate the massive Quantum Kernel Matrix and fits the SVC.
```bash
python Run_QSVC_Legit.py
```

### 3. `Run_QSVC_Mixed_Variance.py` (The Robustness Validator)
Used to prove statistical significance. It mixes the OOD DatasetNinja and the Training CornWeed dataset, expands the quantum kernel to a mathematically intensive 100x100 matrix, and runs the entire training and testing pipeline 10 separate times with different random seeds. It saves every model to `saved_models_variance/`.
```bash
python Run_QSVC_Mixed_Variance.py
```

### 4. `Advanced_Evaluation_Plots.py` (The Analytics Engine)
This script parses the saved models and runs an extensive test-set inference pass. It is responsible for generating all the ROC curves, PR curves, Correlation Heatmaps, and SHAP visual importance overlays found in `benchmark_sota/`.
```bash
python Advanced_Evaluation_Plots.py
```

---

## 🛠️ Requirements & Environment
* **Frameworks**: `PyTorch`, `PennyLane` (Quantum Simulation), `Scikit-Learn`, `Ultralytics` (YOLO)
* **Backend**: PennyLane's `lightning.qubit` plugin is strictly required for the Kernel computation. Attempting to use `default.qubit` will result in unacceptable $O(N^2)$ bottlenecks.
* **Compute**: The QSVC script defaults to `n_jobs=-1` to utilize 100% of available CPU Threads for parallel circuit simulation.

