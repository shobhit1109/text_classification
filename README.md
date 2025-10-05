# LSTM Text Classification / Sentiment Analysis

This repository contains Jupyter notebooks demonstrating LSTM-based text classification and sentiment analysis models. The notebooks are intended for learning and experimentation.

Contents
- `LSTM_text_classification.ipynb` — Text classification with LSTM (full example).

Overview
- Notebooks use TensorFlow / Keras for model building and standard data-science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) for preprocessing, visualization, and evaluation.
Some notebooks may include a small top cell that installs `tensorflow` and `keras` into the notebook kernel's Python interpreter if they are missing. If you run such an installer cell, restart the kernel and re-run cells from the top.

Why this is useful
- Learn the end-to-end NLP pipeline: raw text -> tokenization -> sequence padding -> model input.
- Explore LSTM internals and how recurrent layers process sequences compared with simpler baselines.
- Practice model design choices (embedding size, LSTM units, dropout) and evaluation (train/val split, confusion matrix, precision/recall).
- See visualization and debugging patterns for training curves and prediction errors.
- Easily adapt the notebooks for other text tasks (multi-class classification, sequence tagging) or for production export (SavedModel, ONNX).

# LSTM Text Classification / Sentiment Analysis

Small, hands-on Jupyter notebooks that demonstrate LSTM-based text classification and sentiment analysis using TensorFlow / Keras.

<!-- Badges (replace with real links if desired) -->
[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![tensorflow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](#)

## Table of contents
- [Contents](#contents)
- [Overview](#overview)
- [Why this is useful](#why-this-is-useful)
- [Sample dataset & expected outputs](#sample-dataset--expected-outputs)
- [Recommended environment](#recommended-environment)
- [Pinned (tested) requirements](#pinned-tested-requirements)
- [Quick run](#quick-run-jupyterlab--vs-code)
- [Troubleshooting](#troubleshooting)
- [Notes & recommendations](#notes--recommendations)
- [Contributing](#contributing)

## Contents
- `LSTM_text_classification.ipynb` — Text classification with LSTM (full example).

## Overview
- Notebooks use TensorFlow / Keras for model building and common data-science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) for preprocessing, visualization, and evaluation.
- Some notebooks may include a small top cell that installs `tensorflow` and `keras` into the notebook kernel's Python interpreter if they are missing. If you run such an installer cell, restart the kernel and re-run cells from the top.

## Why this is useful
- Learn the end-to-end NLP pipeline: raw text → tokenization → sequence padding → model input.
- Explore LSTM internals and how recurrent layers process sequences compared with simpler baselines.
- Practice model design choices (embedding size, LSTM units, dropout) and evaluation (train/val split, confusion matrix, precision/recall).
- See visualization and debugging patterns for training curves and prediction errors.
- Easily adapt the notebooks for other text tasks (multi-class classification, sequence tagging) or for production export (SavedModel, ONNX).

## Sample dataset & expected outputs
Below is a short sample dataset (CSV format) that matches the input shape expected by the notebooks.

### Sample dataset (CSV)
```csv
text,label
"I love this movie",1
"This was a terrible product",0
"Absolutely fantastic",1
"Not my taste at all",0
```

Notes
- Each row contains a text string and a binary label (1 = positive, 0 = negative). Replace or extend this CSV with your own data.
- Load the CSV with your preferred method (for example, with pandas) and follow the notebook steps: tokenize, convert to sequences, pad to a fixed length, then train an LSTM-based classifier.

### Expected outputs (small example)
- Training logs: per-epoch loss and accuracy printed by the training loop.
- Example (illustrative) predictions for the sample CSV:
  - "I love this movie"  → probability ≈ 0.85 (positive)
  - "This was a terrible product" → probability ≈ 0.10 (negative)
  - "Absolutely fantastic" → probability ≈ 0.90 (positive)
  - "Not my taste at all" → probability ≈ 0.15 (negative)

These numbers are illustrative; actual probabilities depend on model initialization, hyperparameters, and dataset size.

## Recommended environment
- Python 3.9–3.11 (use a virtual environment).
- 8+ GB of disk and moderate RAM. GPU support is optional; see TensorFlow docs for GPU setup.

## Pinned (tested) requirements
The following versions were observed when the environment was prepared while developing these notebooks:

```
tensorflow==2.20.0
keras==3.11.3
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyterlab
```

## Quick run (JupyterLab / VS Code)
- From PowerShell in this folder, launch JupyterLab:

```powershell
jupyter lab
```

- Or open the folder in VS Code and open the `.ipynb` file using the notebook editor. Run cells top-to-bottom.

## Troubleshooting
- ModuleNotFoundError (e.g., "No module named 'keras'")
  - Ensure you installed packages into the same Python interpreter your notebook kernel is using. From PowerShell:

```powershell
python -m pip install tensorflow keras
```

  - After installing, restart the kernel and re-run cells from the top.

- If pip fails due to permissions, run PowerShell as Administrator or use a virtual environment.

## Notes & recommendations
- Prefer using `tf.keras` (import TensorFlow and use `tf.keras`) to avoid ambiguity between the standalone `keras` package and TensorFlow's built-in Keras. Consider updating notebook imports to `tensorflow.keras` for consistency.
- To capture the exact environment for reproducibility, after installation run:

```powershell
python -m pip freeze > requirements.txt
```

## Contributing
- Suggestions and contributions are welcome. If you want specific changes (add `requirements.txt`, add a sample CSV file, or convert imports to `tensorflow.keras`), tell me and I can apply them.
