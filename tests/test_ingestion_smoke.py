from rag_core.ingest import data_loader

def test_docs_not_empty():

    docs = data_loader()
    assert len(docs) > 0