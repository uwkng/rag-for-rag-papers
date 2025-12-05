from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough

from ingest import data_loader, chunking
from embed import embedding_function
from vectorstore import setup_vectorstore

docs = data_loader()
chunks = chunking(docs)
embedding_model = embedding_function()
vectorstore = setup_vectorstore(docs, chunks, embedding_model)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

chat_model = ChatHuggingFace(llm=llm)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 5}
)

message = """

You are an AI assistant for a Retrieval-Augmented Generation (RAG) application.

Your task:
- Answer the user's question using ONLY the information in the provided context.
- If the context does not contain the answer, say "I don't know based on the provided context."
- Be precise, factual and concise.
- Do not hallucinate or rely on outside knowledge.
- If useful, quote or summarize key parts of the retrieved documents.

Context:
{context}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
             | prompt_template
             | chat_model)

user_input = input("Ask something about RAG: ")
message = rag_chain.invoke(user_input)
print(message.content)