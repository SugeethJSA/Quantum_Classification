import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

def main():
    os.makedirs('benchmark_sota', exist_ok=True)

    # 1. Hardcoded SOTA metrics from the user's terminal output since the previous script crashed before saving them
    sota_data = [
        {"Model": "ResNet18", "OOD_Accuracy": 0.9724, "OOD_Precision": 0.9796, "OOD_Recall": 0.9650, "OOD_F1": 0.9723, "OOD_AUC": 0.9954, "FPS": 58.6},
        {"Model": "MobileNetV3", "OOD_Accuracy": 0.9879, "OOD_Precision": 0.9914, "OOD_Recall": 0.9844, "OOD_F1": 0.9879, "OOD_AUC": 0.9966, "FPS": 64.8},
        {"Model": "EfficientNet-B0", "OOD_Accuracy": 0.9896, "OOD_Precision": 0.9896, "OOD_Recall": 0.9896, "OOD_F1": 0.9896, "OOD_AUC": 0.9969, "FPS": 57.3},
        {"Model": "YOLOv8n-cls", "OOD_Accuracy": 0.9858, "OOD_Precision": 0.9858, "OOD_Recall": 0.9858, "OOD_F1": 0.9858, "OOD_AUC": 0.9858, "FPS": 49.1},
        {"Model": "Custom_Legacy_CNN", "OOD_Accuracy": 0.8799, "OOD_Precision": 0.8810, "OOD_Recall": 0.8790, "OOD_F1": 0.8800, "OOD_AUC": 0.9459, "FPS": 57.5},
        {"Model": "Custom_Legacy_QNN", "OOD_Accuracy": 0.5313, "OOD_Precision": 0.8366, "OOD_Recall": 0.0800, "OOD_F1": 0.1461, "OOD_AUC": 0.7677, "FPS": 61.3},
        {"Model": "Hybrid CNN-QNN (From Scratch)", "OOD_Accuracy": 0.9249, "OOD_Precision": 0.9264, "OOD_Recall": 0.9233, "OOD_F1": 0.9249, "OOD_AUC": 0.9675, "FPS": 55.6},
        {"Model": "Hybrid CNN-QNN (Transfer Learning)", "OOD_Accuracy": 0.9239, "OOD_Precision": 0.9022, "OOD_Recall": 0.9512, "OOD_F1": 0.9261, "OOD_AUC": 0.9784, "FPS": 56.1}
    ]

    df_master = pd.DataFrame(sota_data)

    # 2. Extract standard QSVC from final_benchmark_metrics.csv
    if os.path.exists('final_benchmark_metrics.csv'):
        df_final = pd.read_csv('final_benchmark_metrics.csv')
        # Grab only the rows from the 10-run loop (Legit Run 1 through 10)
        qsvc_standard = df_final[df_final['Model'].str.contains(r'Hybrid CNN-QSVC \(Legit Run', regex=True)]
        if not qsvc_standard.empty:
            avg_qsvc = qsvc_standard.mean(numeric_only=True).to_dict()
            avg_qsvc['Model'] = 'Hybrid CNN-QSVC (Standard 10-Run Avg)'
            avg_qsvc['FPS'] = np.nan
            df_master = pd.concat([df_master, pd.DataFrame([avg_qsvc])], ignore_index=True)

    # 3. Extract Mixed QSVC from variance_mixed_benchmark_metrics.csv
    if os.path.exists('variance_mixed_benchmark_metrics.csv'):
        df_mixed = pd.read_csv('variance_mixed_benchmark_metrics.csv')
        qsvc_mixed = df_mixed[df_mixed['Model'].str.contains(r'Mixed Dataset QSVC', regex=True)]
        if not qsvc_mixed.empty:
            avg_mixed = qsvc_mixed.mean(numeric_only=True).to_dict()
            avg_mixed['Model'] = 'Hybrid CNN-QSVC (Mixed 10-Run Avg)'
            avg_mixed['FPS'] = np.nan
            df_master = pd.concat([df_master, pd.DataFrame([avg_mixed])], ignore_index=True)

    # 4. Save Master CSV
    master_csv_path = os.path.join('benchmark_sota', 'MASTER_BENCHMARK_RESULTS.csv')
    df_master.to_csv(master_csv_path, index=False)
    print(f"Saved consolidated metrics to {master_csv_path}")

    # 5. Generate Bar Plots
    metrics_to_plot = ['OOD_Accuracy', 'OOD_F1', 'OOD_AUC']
    
    # Sort for barplot aesthetics
    df_plot = df_master.sort_values(by='OOD_Accuracy', ascending=False)
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Color code: Classical = Grey, Quantum = Purple
        colors = ['#9b59b6' if ('QNN' in m or 'QSVC' in m) else '#95a5a6' for m in df_plot['Model']]
        
        sns.barplot(x=metric, y='Model', data=df_plot, hue='Model', palette=colors, legend=False)
        plt.title(f'Model Comparison: {metric}')
        plt.xlim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join('benchmark_sota', f'BarPlot_{metric}.png'), dpi=300)
        plt.close()
        print(f"Generated BarPlot_{metric}.png")

    # 6. Generate Radar Chart
    def make_radar_chart(df, filename):
        categories = ['OOD_Accuracy', 'OOD_Precision', 'OOD_Recall', 'OOD_F1', 'OOD_AUC']
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories, color='grey', size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
        plt.ylim(0, 1.05)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        
        for i, (idx, row) in enumerate(df.iterrows()):
            values = row[categories].values.flatten().tolist()
            values += values[:1]
            
            # Highlight quantum models
            if "QNN" in row['Model'] or "QSVC" in row['Model']:
                linewidth = 4
                linestyle = 'solid'
            else:
                linewidth = 1.5
                linestyle = 'dashed'
                
            ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle, label=row['Model'], color=colors[i])
            if "QNN" in row['Model'] or "QSVC" in row['Model']:
                ax.fill(angles, values, color=colors[i], alpha=0.15)
                
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
        plt.title("SOTA vs Quantum Hybrid Radar Chart", size=18, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join('benchmark_sota', filename), dpi=300, bbox_inches='tight')
        plt.close()

    # Create All-Model Radar
    make_radar_chart(df_master, 'RadarChart_AllModels.png')
    print("Generated RadarChart_AllModels.png")

    # Create Highlight Radar (Top SOTA vs QSVCs)
    best_sota = df_master[~df_master['Model'].str.contains('QNN|QSVC')].sort_values('OOD_Accuracy', ascending=False).head(1)
    best_quantum = df_master[df_master['Model'].str.contains('QSVC')]
    df_highlight = pd.concat([best_sota, best_quantum])
    make_radar_chart(df_highlight, 'RadarChart_Highlights.png')
    print("Generated RadarChart_Highlights.png")

    print("\nAll final graphs and consolidated CSVs have been saved to the benchmark_sota/ directory!")

if __name__ == '__main__':
    main()
