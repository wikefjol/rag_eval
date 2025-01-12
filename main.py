from config import Config
from modules import models, preprocess, vectorstore, rag_chain, knowledge_base
import os
import pickle

def main():
    # Load configuration
    config = Config()

    # Initialize LLM and embeddings
    rag_llm = models.init_llm(config.rag_model_name)
    eval_llm = models.init_llm(config.eval_model_name)
    rag_emb = models.init_emb(config.rag_model_name)
    eval_emb = models.init_emb(config.eval_model_name)

    # Load or preprocess data
    if os.path.exists(config.processed_data_path):
        with open(config.processed_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = preprocess.read_files(config.file_path)
        with open(config.processed_data_path, "wb") as f:
            pickle.dump(data, f)

    # Load or initialize vector store
    vec_store = vectorstore.init_vectorstore(data, eval_emb, config.vectorstore_path)

    # Initialize retriever and build RAG chain
    retriever = vectorstore.init_retriever(vec_store, config.num_most_similar_docs)
    rag_chain = rag_chain.build_rag_chain(retriever, rag_llm)

    # Create knowledge base
    gpt_knowledge_base = knowledge_base.create_knowledge_base(data, config.columns)
    
    # Generate testset and evaluate
    testset = knowledge_base.generate_testset(gpt_knowledge_base, config.num_questions, config.language)
    report = knowledge_base.evaluate(rag_chain, testset, gpt_knowledge_base)

    print("Evaluation Report:", report)

if __name__ == "__main__":
    main()
