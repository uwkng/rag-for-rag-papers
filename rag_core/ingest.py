from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

def data_loader():
    """
    Data Loader to iterate over the PDFs, extract the text and append it to an shared list.
    """
    ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT / "data" / "raw"

    all_docs = []

    for pdf in DATA_PATH.iterdir():

        loader = PyPDFLoader(
            str(pdf),
            mode="page",
            images_inner_format="markdown-img",
            images_parser=RapidOCRBlobParser(),
        )
        
        all_docs.extend(loader.load())
        
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