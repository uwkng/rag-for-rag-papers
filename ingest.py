from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from pypdf import PdfReader

DATA_PATH = Path("data/raw")

def data_loader():
    """
    Data Loader to iterate over the PDFs, extract the text and append it to an shared list.
    """

    all_docs = []

    for pdf in DATA_PATH.iterdir():
        reader = PdfReader(str(pdf))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        all_docs.append(text)
        return all_docs

def chunking(all_docs):
    """
    Creates chunks from an list.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    full_text = "\n\n".join(all_docs)
    chunked_texts = splitter.split_text(full_text)
    return chunked_texts