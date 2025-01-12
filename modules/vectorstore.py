import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

def init_vectorstore(df, embedding, persist_directory):
    if os.path.exists(persist_directory):
        return Chroma(embedding_function=embedding, persist_directory=persist_directory)
    
    chunks = []
    sem_text_splitter = SemanticChunker(embedding)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text_chunks = sem_text_splitter.create_documents([row["text"]])
        chunks.extend(text_chunks)
    vec_store = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    vec_store.add_documents(chunks)
    return vec_store

def init_retriever(vec_store, k):
    return vec_store.as_retriever(search_kwargs={"k": k})
