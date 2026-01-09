from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

def data_loader():
    """
    Data Loader to iterate over the PDFs, extract the text and append it to an shared list.
    """
    ROOT = Path(__file__).resolve().parents[2]
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
        chunk_size = 600,
        chunk_overlap = 100,
        length_function=len,
        is_separator_regex=False,
    )

    chunked = splitter.split_documents([all_docs])
    return chunked