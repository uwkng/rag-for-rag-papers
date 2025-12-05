from langchain_chroma import Chroma

from ingest import data_loader, chunking
from embed import embedding_function

all_docs = data_loader()
chunked_docs = chunking(all_docs)
embedding_func = embedding_function()

vector_store = Chroma(
    chunked_docs,
    embedding_function=embedding_func,
    persist_directory="chroma_db"
)

results = vector_store.similarity_search(query="artifical intelligence", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")