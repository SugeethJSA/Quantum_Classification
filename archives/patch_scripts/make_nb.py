import json

with open("hybrid_QC_CNN_v2.py", "r") as f:
    code = f.read()

cells = []
blocks = code.split("\n\n")
for block in blocks:
    if not block.strip(): continue
    lines = block.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("hybrid_QC_CNN_v2_Fixed.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully as hybrid_QC_CNN_v2_Fixed.ipynb")
