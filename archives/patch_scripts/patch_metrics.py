import json
import codecs

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the metrics extraction
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update the metrics calculation
        if 'acc = accuracy_score' in source and 'f1 = f1_score' in source:
            new_source = source.replace(
                "    f1 = f1_score(res['labels'], res['preds'])",
                "    f1 = f1_score(res['labels'], res['preds'])\n    prec = precision_score(res['labels'], res['preds'], zero_division=0)\n    rec = recall_score(res['labels'], res['preds'], zero_division=0)"
            )
            new_source = new_source.replace(
                "        'Accuracy': acc,",
                "        'Accuracy': acc,\n        'Precision': prec,\n        'Recall': rec,"
            )
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# Find where to insert the new graph (after the table display, before core comparison graphs)
table_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'display(df)' in "".join(cell['source']):
        table_cell_idx = i
        break

if table_cell_idx != -1:
    new_markdown_cell = {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1.5. Performance Metrics Comparison\n",
        "Grouped bar chart showing Accuracy, Precision, Recall, and F1 Score for all models."
      ]
    }
    
    new_code_cell = {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_melted = df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'], \n",
        "                    var_name='Metric', value_name='Score')\n",
        "plt.figure(figsize=(12, 6))\n",
        "sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Set2')\n",
        "plt.title('Performance Metrics: Accuracy, Precision, Recall, and F1 Score')\n",
        "plt.ylim(0, 1.1)\n",
        "plt.legend(loc='lower right')\n",
        "plt.savefig('performance_metrics_grouped.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()"
      ]
    }
    
    # Check if we already inserted it
    inserted = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown' and '## 1.5. Performance Metrics' in "".join(cell['source']):
            inserted = True
            break
            
    if not inserted:
        nb['cells'].insert(table_cell_idx + 1, new_markdown_cell)
        nb['cells'].insert(table_cell_idx + 2, new_code_cell)

with codecs.open('Benchmark_Multiple_SOTA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Added Precision, Recall, and a Grouped Bar Chart to the notebook!")
