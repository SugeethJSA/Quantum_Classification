import json

with open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Add %matplotlib inline
        if 'import matplotlib.pyplot as plt' in source and '%matplotlib inline' not in source:
            source = '%matplotlib inline\n' + source
            
        # Fix inplace ReLU for SHAP
        if 'resnet = models.resnet18' in source:
            fix = "\n# Fix inplace ReLU for SHAP\nfor m_resnet in resnet.modules():\n    if hasattr(m_resnet, 'inplace'):\n        m_resnet.inplace = False\n"
            if '# Fix inplace ReLU for SHAP' not in source:
                source = source.replace('resnet = resnet.to(device)', 'resnet = resnet.to(device)' + fix)
        
        # Save plots
        if 'plt.show()' in source:
            if 'Model Accuracy Comparison' in source:
                source = source.replace('plt.show()', 'plt.savefig("core_comparison.png", dpi=300, bbox_inches="tight")\nplt.show()')
            elif 'Radar Comparison' in source:
                source = source.replace('plt.show()', 'plt.savefig("radar_chart.png", dpi=300, bbox_inches="tight")\nplt.show()')
            elif 'ROC Curves' in source:
                source = source.replace('plt.show()', 'plt.savefig("roc_pr_curves.png", dpi=300, bbox_inches="tight")\nplt.show()')
            elif 'Confusion Matrix' in source:
                source = source.replace('plt.show()', 'plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")\nplt.show()')
            elif 'Correlation Heatmap' in source:
                source = source.replace('plt.show()', 'plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")\nplt.show()')
        
        # SHAP save
        if 'shap.image_plot' in source:
            if 'plt.savefig' not in source:
                source = source.replace('shap.image_plot(shap_numpy, -test_numpy)', 
                                        'shap.image_plot(shap_numpy, -test_numpy, show=False)\nimport matplotlib.pyplot as plt\nplt.savefig("shap_values.png", dpi=300, bbox_inches="tight")\nplt.show()')

        # Put back
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print("Notebook patched for SHAP and image saving!")
