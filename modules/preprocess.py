import os
import pdfplumber
import pandas as pd
import pickle
import re
from uuid import uuid4


def clean_text(text, remove_newlines=True, remove_extra_spaces=True):
    """
    Cleans text by removing unwanted characters.

    Parameters:
        text (str): The input text.
        remove_newlines (bool): Whether to remove newline characters.
        remove_extra_spaces (bool): Whether to collapse multiple spaces into one.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    if remove_newlines:
        text = re.sub(r"[\\n\\t\\r]+", " ", text)
    if remove_extra_spaces:
        text = re.sub(r"\\s+", " ", text)
    return text.strip()


def read_files(filepath, file_extension=".pdf"):
    """
    Reads files from the specified directory and extracts text from PDFs.

    Parameters:
        filepath (str): Path to the directory containing files.
        file_extension (str): File extension to filter by (default is ".pdf").

    Returns:
        pd.DataFrame: DataFrame containing extracted text, file names, and page numbers.
    """
    all_data = []
    for dirpath, _, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.endswith(file_extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            all_data.append({
                                "file_name": filename,
                                "page_number": page.page_number,
                                "text": clean_text(text),
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(all_data)


def load_processed_data(processed_data_path):
    """
    Loads processed data if it exists.

    Parameters:
        processed_data_path (str): Path to the processed data pickle file.

    Returns:
        DataFrame or None: Loaded data or None if the file doesn't exist.
    """
    if os.path.exists(processed_data_path):
        try:
            print(f"Loading processed data from {processed_data_path}...")
            with open(processed_data_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading processed data: {e}")
    return None


def preprocess_and_save(raw_data_path, processed_data_path):
    """
    Preprocesses raw data and saves it to the processed data path.

    Parameters:
        raw_data_path (str): Path to the raw data directory.
        processed_data_path (str): Path to save the processed data.

    Returns:
        DataFrame: The processed data.
    """
    print(f"Preprocessing raw data from {raw_data_path}...")
    data = read_files(raw_data_path)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    print(f"Saving processed data to {processed_data_path}...")
    with open(processed_data_path, "wb") as f:
        pickle.dump(data, f)
    return data


def prepare_data(processed_data_path, raw_data_path):
    """
    Handles loading or preprocessing data and ensures it is saved for future use.

    Parameters:
        processed_data_path (str): Path to the processed data pickle file.
        raw_data_path (str): Path to the raw data directory.

    Returns:
        DataFrame: Loaded or processed data.
    """
    data = load_processed_data(processed_data_path)
    if data is None:
        data = preprocess_and_save(raw_data_path, processed_data_path)
    return data
