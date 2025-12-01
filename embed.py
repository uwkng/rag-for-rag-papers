from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embedding_model_name = "BAAI/bge-small-en"
embedding_model_kwargs = {"device: cpu"}
embedding_model_encode_kwargs = {"normalize_embeddings": True}

embedding_model = HuggingFaceBgeEmbeddings(
    model_name = embedding_model_name,
    model_kwargs = embedding_model_kwargs,
    encode_kwargs = embedding_model_encode_kwargs,
)