# Load & Chunk PDFs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from pypdf import PdfReader

DATA_PATH = Path("data/raw")

def data_loader():
    """
    
    """
    all_docs = []

    for pdf in DATA_PATH.iterdir():
        reader = PdfReader(str(pdf))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        all_docs.append(text)
        return all_docs