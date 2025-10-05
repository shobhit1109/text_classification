# LSTM Text Classification / Sentiment Analysis

This repository contains Jupyter notebooks demonstrating LSTM-based text classification and sentiment analysis models. The notebooks are designed for educational experiments and include convenience code to install required packages when missing.

Included files
- `LSTM_sentiment_analysis.ipynb` — LSTM demo for sentiment analysis (imports and example model).
- `LSTM_text_classification.ipynb` — Text classification with LSTM (longer notebook).

Quick overview
- The notebooks use TensorFlow / Keras for model building, and typical data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) for preprocessing, visualization, and evaluation.
- A small helper cell at the top of `LSTM_sentiment_analysis.ipynb` ensures TensorFlow/Keras are installed into the notebook kernel's Python environment when you run it.

Recommended environment
- Python 3.9–3.11 (use a virtual environment).
- At least 8 GB of disk and moderate RAM. GPU support is optional — if you want GPU acceleration, install the GPU-enabled TensorFlow following NVIDIA/CUDA compatibility instructions.

Setup (recommended)
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install the common dependencies (CPU TensorFlow):

```powershell
python -m pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn jupyterlab
```

If you prefer to let the notebook install missing packages automatically, open the notebook and run the first cell (it installs `tensorflow` and `keras` into the kernel's interpreter). After installation, restart the kernel and run the rest of the notebook.

How to open and run the notebooks
- From VS Code: open the folder, click the notebook (`.ipynb`) file, and use the built-in notebook UI. Run cells from top to bottom. If the top cell installs packages, restart the kernel after it finishes.
- From JupyterLab / Notebook: launch `jupyter lab` or `jupyter notebook` in this folder and open the notebook files.

Troubleshooting
- ModuleNotFoundError (e.g., "No module named 'keras'")
  - If you see this while running a notebook, run the top installer cell (if present) or install packages into the same Python interpreter your notebook kernel uses.
  - To install from PowerShell into the same Python used by the notebook:

```powershell
python -m pip install tensorflow keras
```

  - After installing packages, restart the kernel and re-run the notebook cells from the top.

- If installations fail due to OS or permissions, consider running PowerShell as Administrator or using a virtual environment.

Notes and recommendations
- Use `tf.keras` (from `import tensorflow as tf` then `tf.keras`) to avoid ambiguity between the standalone `keras` package and `tf.keras`. The notebooks currently import `keras` in places; consider replacing those imports with `tensorflow.keras` for consistency.
- If you want to capture the exact environment for reproducibility, create a `requirements.txt` after installing packages:

```powershell
python -m pip freeze > requirements.txt
```

- For GPU support, follow official TensorFlow docs for matching CUDA/cuDNN versions and drivers.

Next steps I can do for you
- Add a `requirements.txt` with tested versions (I can generate this from the notebook run: example versions used were `tensorflow==2.20.0`, `keras==3.11.3`).
- Update notebooks to consistently use `tensorflow.keras` imports.
- Add a small sample dataset and a short README section showing expected notebook outputs.

If you'd like any of those, tell me which and I'll make the changes.