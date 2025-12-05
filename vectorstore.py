from langchain_chroma import Chroma

def setup_vectorstore(all_docs, chunked_docs, embedding_func):
    """
    Sets up the vectorstore.
    """
    vector_store = Chroma.from_texts(
        texts = chunked_docs,
        embedding = embedding_func,
        persist_directory="chroma_db"
    )

    return vector_store