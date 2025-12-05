from langchain_chroma import Chroma

from ingest import data_loader, chunking
from embed import embedding_function

all_docs = data_loader()
chunked_docs = chunking(all_docs)
embedding_func = embedding_function()

vector_store = Chroma.from_texts(
    texts = chunked_docs,
    embedding = embedding_func,
    persist_directory="chroma_db"
)