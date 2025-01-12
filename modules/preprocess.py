import os
import pdfplumber
import pandas as pd
from uuid import uuid4
import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[\\n\\t\\r]+", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

def read_files(filepath):
    all_data = []
    for dirpath, _, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.endswith(".pdf"):
                file_path = os.path.join(dirpath, filename)
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        all_data.append({
                            "file_name": filename,
                            "page_number": page.page_number,
                            "text": clean_text(text),
                        })
    return pd.DataFrame(all_data)
