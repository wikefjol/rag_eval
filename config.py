class Config:
    file_path = "./data/raw"
    processed_data_path = "./data/processed/processed_data.pkl"
    vectorstore_path = "./data/vectorstore"
    columns = ["text", "file_name", "page_number"]
    num_most_similar_docs = 5
    num_questions = 10
    language = "en"
    rag_model_name = "ChatGPT3.5-turbo"
    eval_model_name = "ChatGPT3.5-turbo"
