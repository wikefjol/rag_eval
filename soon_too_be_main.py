import logging
from modules import RAG_tools, models, preprocess
from modules.vectorstore import VectorstoreHandler
from modules.testset_manager import save_test_set
from giskard.rag import KnowledgeBase, generate_testset, evaluate
from modules.giskard_wrappers import GiskardEmbeddingAdapter, GiskardLLMAdapter
#from modules.rag_interface import run_rag, create_test_set#, evaluate_rag
from modules.eval_tools import display_evaluation_results

def main():
    # Enable logging
    logging.getLogger().setLevel(logging.WARNING)

    # File paths
    processed_data_path = "data/processed/processed_data.pkl"
    raw_data_path = "data/raw"
    vsts_dir = "data/vectorstores"
    reports_dir = "eval_results"

    # Define LLM and embedding configurations
    rag_model_name = "ChatGPT3.5-turbo"
    rag_emb_name = "hf-mpnet-base-v2"

    kb_model_name = "ChatGPT3.5-turbo"
    kb_emb_name = "hf-mpnet-base-v2"

    k = 3  # Number of documents to retrieve
    columns = ["text", "file_name", "page_number"]

    # Prepare the dataset
    data = preprocess.prepare_data(processed_data_path, raw_data_path)

    # Initialize RAG chain components
    rag_llm = models.init_llm(rag_model_name)
    rag_embedding = models.init_emb(rag_emb_name)

    # Initialize KnowledgeBase components
    kb_llm = models.init_llm(kb_model_name)
    kb_embedding = models.init_emb(kb_emb_name)
    wrapped_llm = GiskardLLMAdapter(kb_llm)
    wrapped_embedding = GiskardEmbeddingAdapter(kb_embedding)

    # Initialize the VectorstoreHandler
    vst_handler = VectorstoreHandler(
        persist_directory=vsts_dir,
        embedding=rag_embedding,
        dataset=data,
        force_rebuild=False  # Set to True to force rebuild of the vectorstore
    )

    # Build vectorstore and retriever
    vst = vst_handler.init_vectorstore()
    retriever = vst_handler.init_retriever(vst, k)
    chain = RAG_tools.build_rag_chain(retriever, rag_llm)
    answer_fn = RAG_tools.create_answer_fn(chain)

    # Create the KnowledgeBase
    kb = KnowledgeBase(
        data=data,
        embedding_model=wrapped_embedding,
        llm_client=wrapped_llm
    )

    # Generate the test set
    test_set = generate_testset(
        knowledge_base=kb,
        num_questions=10,
        language='en',
        agent_description=(
            "This is an agent that uses the following context to answer the question. "
            "It only uses information from the context provided. It does not ask questions."
        )
    )

    # Save the test set
    save_test_set(test_set, kb_model_name, kb_emb_name, data)


    report = evaluate(answer_fn=answer_fn, testset=test_set, knowledge_base=kb)
    report.save(reports_dir)
    display_evaluation_results(report)



# def main():
#     # Enable logging
#     logging.getLogger().setLevel(logging.WARNING)

#     while True:
#         print("\nWelcome to the RAG Project Interface!")
#         print("1. Run a RAG chain")
#         print("2. Create a new test set")
#         print("3. Evaluate a RAG chain against a test set (NOT IMPLEMENTED YET)")
#         print("4. Exit")

#         choice = input("Enter the number of your choice: ").strip()

#         if choice == "1":
#             run_rag()
#         elif choice == "2":
#             create_test_set()
#         elif choice == "3":

#             evaluate_rag()
#         elif choice == "4":
#             print("Goodbye!")
#             break
#         else:
#             print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
