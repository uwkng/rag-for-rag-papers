from langchain_huggingface import HuggingFaceEmbeddings

def embedding_function():

    model = "Qwen/Qwen3-Embedding-0.6B"

    embedding_model = HuggingFaceEmbeddings(
        model_name=model
    )

    return embedding_model