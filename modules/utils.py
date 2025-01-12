import hashlib
import os
from datetime import datetime

def compute_dataset_hash(data):
    """
    Compute a unique hash for the dataset.
    """
    concatenated_text = ''.join(data['text'].tolist())
    return hashlib.sha256(concatenated_text.encode('utf-8')).hexdigest()

def generate_test_set_path(base_dir, model_name, embedding_name, dataset_hash):
    """
    Generate a directory and file path for the test set.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    subdir = os.path.join(base_dir, model_name, embedding_name, dataset_hash)
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"test_set_{timestamp}.jsonl")