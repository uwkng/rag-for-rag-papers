from langchain_chroma import Chroma

from pathlib import Path

PERSIST_DIR = "chroma_db"
TEXTS = 



chroma_path = Path("./chroma_db")
chroma_directory = chroma_path.iterdir()
chroma_files_existing = any(e.is_file() for e in chroma_directory)

def make_vectorstore_available(all_docs, chunked_docs, embedding_func):
    """
    Sets up the vectorstore. Creates either the vectorstore from ground up or loads the existing one, checks automatically.
    """


    vector_store = Chroma.from_texts(
        texts = chunked_docs,
        embedding = embedding_func,
        persist_directory="chroma_db"
    )

    return vector_store

