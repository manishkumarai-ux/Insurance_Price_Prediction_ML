import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    "data/insurance.csv",
    "notebooks/Insurance_Price_Prediction.ipynb",
    "src/data_preprocessing.py",
    "src/train_model.py",
    "src/evaluate_model.py",
    "src/predict.py",
    "models/best_model.pkl",
    "images/charges_distribution.png",
    "images/correlation_heatmap.png",
    "images/model_comparison.png",
    "requirements.txt",
    "README.md",
    "LICENSE",
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)
    try:

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory: {filedir}")

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, 'w') as f:
                pass
            logging.info(f"Created file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")
    except Exception as e:
        logging.error(f"Error creating {filepath}: {e}")