import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai.embeddings import OpenAIEmbeddings
import torch

AVAILABLE_LLMS = {
    "ChatGPT4o": "gpt-4o",
    "ChatGPT3.5-turbo": "gpt-3.5-turbo",
    "Llama3.2-3b": "llama3.2:3b",
}

AVAILABLE_EMBS = {
    "ChatGPT4o": "openai",
    "ChatGPT3.5-turbo": "openai",
    "Llama3.2-3b": "sentence-transformers/all-mpnet-base-v2",
}

def init_llm(model_name):
    if model_name.startswith("ChatGPT"):
        return ChatOpenAI(model=AVAILABLE_LLMS[model_name])
    elif model_name.startswith("Llama"):
        return OllamaLLM(model=AVAILABLE_LLMS[model_name])
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

def init_emb(model_name):
    if model_name.startswith("ChatGPT"):
        return OpenAIEmbeddings()
    elif model_name.startswith("Llama"):
        return HuggingFaceEmbeddings(model_name=AVAILABLE_EMBS[model_name])
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")
