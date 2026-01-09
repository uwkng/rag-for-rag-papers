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
        chunk_size = 1000,
        chunk_overlap = 200
    )

    full_text = "\n\n".join(all_docs)
    chunked_texts = splitter.split_text(full_text)
    return chunked_texts

def smoke_test_ingestion():
    """
    Smoke tests to ensure that text, tables and images get retrieved
    """
    print("---INGESTION---")
    docs = data_loader()
    print(f"Length of all the ingested documents: {len(docs)}")
    print(f"PDF Site with figure: {docs[2]}")
    print(f"PDF Site with table: {docs[24]}")


# Smoke tests to ensure that text, tables and images get retrieved
if __name__ == "__main__":
    smoke_test_ingestion()