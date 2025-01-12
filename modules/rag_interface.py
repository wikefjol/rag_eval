
from modules.vectorstore import VectorstoreHandler
from modules.RAG_tools import build_rag_chain, handle_query, create_answer_fn
from modules.preprocess import prepare_data
from modules.models import init_llm, init_emb, AVAILABLE_LLMS, AVAILABLE_EMBS
from modules.testset_manager import save_test_set
from modules.giskard_wrappers import GiskardLLMAdapter, GiskardEmbeddingAdapter

from giskard.rag import generate_testset, KnowledgeBase, QATestset, evaluate
import json
import os
def run_rag():
    print("\n--- Run a RAG Chain ---")
    
    # Display available LLMs
    print("Available LLMs:")
    llm_choices = list(AVAILABLE_LLMS.keys())
    for i, llm_name in enumerate(llm_choices, 1):
        print(f"{i}. {llm_name}")
    llm_index = int(input("Enter the number of the LLM to use: ")) - 1
    llm_name = llm_choices[llm_index]
    
    # Display available embedding models
    print("Available Embedding Models:")
    emb_choices = list(AVAILABLE_EMBS.keys())
    for i, emb_name in enumerate(emb_choices, 1):
        print(f"{i}. {emb_name}")
    emb_index = int(input("Enter the number of the embedding model to use: ")) - 1
    emb_name = emb_choices[emb_index]
    
    # Get dataset path
    raw_data_path = input("Enter the dataset path (default: data/raw): ").strip() or "data/raw"
    processed_data_path = "data/processed/processed_data.pkl"
    vsts_dir = "data/vectorstores"

    # Prepare dataset
    data = prepare_data(processed_data_path, raw_data_path)

    # Initialize models
    llm = init_llm(llm_name)
    embedding = init_emb(emb_name)

    # Setup vectorstore and retriever
    handler = VectorstoreHandler(vsts_dir, embedding, data)
    vectorstore = handler.init_vectorstore()
    retriever = handler.init_retriever(vectorstore, k=3)

    # Build RAG chain
    rag_chain = build_rag_chain(retriever, llm)

    # RAG loop
    print("\nEntering RAG query mode. Type 'exit' or 'q' to return to the main menu.")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in {"q", "exit"}:
            print("\nExiting RAG query mode.")
            break

        result = handle_query(rag_chain, query)
        print(f"\nQuery: {result['question']}\nAnswer: {result['answer']}")
        print("\nRetrieved documents:")
        for doc in result["docs"]:
            doc_id = doc.metadata.get("id", "N/A")
            content = doc.page_content[:300]  # Show first 300 characters for brevity
            print(f"- Document ID: {doc_id}")
            print(f"  Content: {content}\n")

def create_test_set():
    print("\n--- Create a Test Set ---")
    print("1. Import Q&A JSON")
    print("2. Create dynamically using an LLM")
    option = input("Enter your choice: ").strip()

    if option == "1":
        # Handle Q&A JSON import
        json_path = input("Enter the path to the Q&A JSON file: ").strip()
        try:
            with open(json_path, "r") as f:
                qa_pairs = json.load(f)
                print(f"Successfully loaded {len(qa_pairs)} Q&A pairs from {json_path}.")
            
            # Convert Q&A JSON into a test set (logic depends on format of JSON)
            test_set = QATestset(qa_pairs)  # Assuming QATestset accepts a list of question-answer pairs
            
            # Save the test set
            model_name = input("Enter model name for test set organization: ").strip()
            emb_name = input("Enter embedding name for test set organization: ").strip()
            dataset_placeholder = "JSON-Source"  # Placeholder as no dataset is tied to this test set
            save_test_set(test_set, model_name, emb_name, dataset_placeholder)
        except Exception as e:
            print(f"Failed to load JSON file: {e}")
    elif option == "2":
        # Dynamic creation using LLM
        print("\n--- Select LLM and Embedding ---")
        print("Available LLMs:")
        llm_choices = list(AVAILABLE_LLMS.keys())
        for i, llm_name in enumerate(llm_choices, 1):
            print(f"{i}. {llm_name}")
        llm_index = int(input("Enter the number of the LLM to use: ")) - 1
        llm_name = llm_choices[llm_index]

        print("\nAvailable Embedding Models:")
        emb_choices = list(AVAILABLE_EMBS.keys())
        for i, emb_name in enumerate(emb_choices, 1):
            print(f"{i}. {emb_name}")
        emb_index = int(input("Enter the number of the embedding model to use: ")) - 1
        emb_name = emb_choices[emb_index]

        # Dataset path
        dataset_path = input("Enter the dataset path (default: data/raw): ").strip() or "data/raw"
        processed_data_path = "data/processed/processed_data.pkl"

        # Prepare dataset
        data = prepare_data(processed_data_path, dataset_path)

        # Initialize LLM and embedding
        llm = init_llm(llm_name)
        embedding = init_emb(emb_name)

        # Wrap models for Giskard
        wrapped_llm = GiskardLLMAdapter(llm)
        wrapped_embedding = GiskardEmbeddingAdapter(embedding)

        # Build KnowledgeBase
        kb = KnowledgeBase(
            data=data,
            embedding_model=wrapped_embedding,
            llm_client=wrapped_llm
        )

        # Generate the test set
        num_questions = int(input("Enter the number of questions to generate (default: 10): ").strip() or 10)
        test_set = generate_testset(
            knowledge_base=kb,
            num_questions=num_questions,
            language="en",
        )

        # Save the test set
        save_test_set(test_set, llm_name, emb_name, data)
    else:
        print("Invalid choice. Returning to main menu.")


def display_test_sets(test_sets, page_size=9):
    """
    Display test sets in a paginated manner and allow the user to select one.

    Args:
        test_sets (list): A list of available test sets, where each entry is a tuple
                          (model, embedding, dataset_hash, file_path).
        page_size (int): Number of test sets to display per page.

    Returns:
        str: Path to the selected test set file, or None if the user cancels.
    """
    if not test_sets:
        print("No test sets available.")
        return None

    total_pages = (len(test_sets) + page_size - 1) // page_size
    current_page = 1

    while True:
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_items = test_sets[start_idx:end_idx]

        print(f"\n--- Available Test Sets (Page {current_page}/{total_pages}) ---")
        for idx, (model, embedding, dataset_hash, file_path) in enumerate(page_items, start=1):
            print(
                f"{idx + start_idx}. Model: {model}, Embedding: {embedding}, "
                f"Dataset Hash: {dataset_hash}, File: {file_path}"
            )

        # Navigation options
        print("\nOptions:")
        if current_page > 1:
            print("p. Previous Page")
        if current_page < total_pages:
            print("n. Next Page")
        print("x. Cancel")

        # User input
        user_input = input("\nEnter the number of the test set to select, or an option: ").strip().lower()

        if user_input == "p" and current_page > 1:
            current_page -= 1
        elif user_input == "n" and current_page < total_pages:
            current_page += 1
        elif user_input == "x":
            print("Cancelled.")
            return None
        elif user_input.isdigit():
            selected_index = int(user_input) - 1
            if 0 <= selected_index < len(test_sets):
                selected_test_set = test_sets[selected_index]
                print(f"Selected Test Set: {selected_test_set[-1]}")
                return selected_test_set[-1]
            else:
                print("Invalid choice. Please try again.")
        else:
            print("Invalid input. Please try again.")

def display_evaluation_results(report):
    """
    Display the evaluation results in a readable format.

    Args:
        report (RAGReport): The report generated by the evaluation.
    """
    print("\n=== RAG Evaluation Results ===")

    # Overall Summary
    print(f"\nOverall Correctness: {report.correctness * 100:.2f}%")
    print(f"Total Test Cases: {len(report._testset)}")

    # Component Scores
    print("\n--- RAG Component Scores ---")
    component_scores = report.component_scores()
    for component, score in component_scores["score"].items():
        print(f"{component}: {score * 100:.2f}%")

    # Correctness by Question Type
    print("\n--- Correctness by Question Type ---")
    question_type_correctness = report.correctness_by_question_type()
    for question_type, row in question_type_correctness.iterrows():
        print(f"{question_type}: {row['correctness'] * 100:.2f}%")

    # Correctness by Topic
    print("\n--- Correctness by Topic ---")
    topic_correctness = report.correctness_by_topic()
    for topic, row in topic_correctness.iterrows():
        print(f"{topic}: {row['correctness'] * 100:.2f}%")

    # Failures
    print("\n--- Problematic Cases ---")
    failures = report.get_failures()
    if not failures.empty:
        for idx, row in failures.iterrows():
            print(f"Question: {row['question']}")
            print(f"Ground Truth: {row['reference_answer']}")
            print(f"Model Response: {row['agent_answer']}")
            print(f"Reason: {row['correctness_reason']}")
            print("-" * 40)
    else:
        print("No failures detected.")

    # Save Report to Disk (Optional)
    save_option = input("\nWould you like to save the report to disk? (y/n): ").strip().lower()
    if save_option == "y":
        folder_path = input("Enter folder path to save the report: ").strip()
        report.save(folder_path)
        print(f"Report saved at {folder_path}")

def list_test_sets(base_dir="data/test_sets"):
    """
    List all available test sets in the structured directory.

    Args:
        base_dir: The base directory for test sets.

    Returns:
        List[Tuple[str, str, str, str]]: List of tuples with model, embedding, dataset hash, and test set file path.
    """
    available_test_sets = []
    for model in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model)
        if os.path.isdir(model_dir):
            for embedding in os.listdir(model_dir):
                embedding_dir = os.path.join(model_dir, embedding)
                if os.path.isdir(embedding_dir):
                    for dataset_hash in os.listdir(embedding_dir):
                        dataset_dir = os.path.join(embedding_dir, dataset_hash)
                        if os.path.isdir(dataset_dir):
                            for test_set_file in os.listdir(dataset_dir):
                                if test_set_file.endswith(".jsonl"):
                                    available_test_sets.append((model, embedding, dataset_hash, os.path.join(dataset_dir, test_set_file)))
    return available_test_sets


def print_rag_report_summary(report):
    """
    Generate and print a summary of the RAGReport.

    Args:
        report (RAGReport): The evaluation report object.
    """
    print("\n--- RAG Evaluation Report Summary ---\n")

    # 1. Overall Performance
    print("1. Overall Performance")
    total_questions = len(report._dataframe)
    overall_correctness = report.correctness * 100
    print(f"   - Total Questions: {total_questions}")
    print(f"   - Correctness: {overall_correctness:.2f}% ({int(overall_correctness / 100 * total_questions)}/{total_questions})\n")

    # 2. Correctness by Question Type
    print("2. Correctness by Question Type")
    question_type_correctness = report.correctness_by_question_type()
    for question_type, row in question_type_correctness.iterrows():
        print(f"   - {question_type}: {row['correctness'] * 100:.2f}%")
    print()

    # 3. Correctness by Topic
    print("3. Correctness by Topic")
    topic_correctness = report.correctness_by_topic()
    for topic, row in topic_correctness.iterrows():
        print(f"   - {topic}: {row['correctness'] * 100:.2f}%")
    print()

    # 4. Failures
    print("4. Failures")
    failures = report.get_failures()
    print(f"   - Total Failures: {len(failures)}")
    if not failures.empty:
        example_failure = failures.iloc[0]
        print("   - Example Failure:")
        print(f"       - Question: \"{example_failure['question']}\"")
        print(f"       - Ground Truth: \"{example_failure['reference_answer']}\"")
        print(f"       - Model Response: \"{example_failure['agent_answer']}\"")
        print(f"       - Reason for Failure: {example_failure['correctness_reason']}")
    print()

    # 5. Component Scores
    print("5. Component Scores")
    component_scores = report.component_scores()
    for component, row in component_scores.iterrows():
        print(f"   - {component}: {row['score'] * 100:.2f}%")
    print()

    # 6. Recommendations
    print("6. Recommendations")
    print(f"   - {report._recommendation}\n")

    # 7. Statistical Overview (Optional)
    print("7. Statistical Overview")
    if report.metric_names:
        print("   - Additional metrics available. Use report.get_metrics_histograms() for more details.")
    else:
        print("   - No additional metrics available.")
    print()

    print("--- End of Summary ---\n")

def evaluate_rag():
    """
    Evaluate a RAG chain against a test set and display results in a user-friendly format.
    """
    print("\n--- Evaluate a RAG Chain ---")
    vsts_dir = "data/vectorstores"
    # Step 1: Select LLM
    print("\nAvailable LLMs:")
    llms = list(AVAILABLE_LLMS.keys())  # Assuming AVAILABLE_LLMS is a dictionary of supported LLMs
    for idx, llm in enumerate(llms, start=1):
        print(f"{idx}. {llm}")
    llm_choice = int(input("Select the LLM to use by number: ").strip())
    if not (1 <= llm_choice <= len(llms)):
        print("Invalid choice. Exiting evaluation.")
        return
    llm_name = llms[llm_choice - 1]

    # Step 2: Select Embedding Model
    print("\nAvailable Embedding Models:")
    embeddings = list(AVAILABLE_EMBS.keys())  # Assuming AVAILABLE_EMBS is a dictionary of supported embeddings
    for idx, emb in enumerate(embeddings, start=1):
        print(f"{idx}. {emb}")
    emb_choice = int(input("Select the embedding model to use by number: ").strip())
    if not (1 <= emb_choice <= len(embeddings)):
        print("Invalid choice. Exiting evaluation.")
        return
    emb_name = embeddings[emb_choice - 1]

    # Step 3: Select Test Set
    print("\nAvailable Test Sets:")
    test_sets = list_test_sets()  # Retrieve available test sets    
    selected_test_set_path = display_test_sets(test_sets)
    if not selected_test_set_path:
        return  # User cancelled or no test sets available

    # Proceed with the selected test set
    test_set = QATestset.load(selected_test_set_path)
    print(f"\nLoaded Test Set: {selected_test_set_path}")
    for idx, (model, embedding, dataset_hash, file_path) in enumerate(test_sets, start=1):
        print(f"{idx}. Model: {model}, Embedding: {embedding}, Dataset Hash: {dataset_hash}, File: {file_path}")
    test_set_choice = int(input("Select the test set to use by number: ").strip())
    if not (1 <= test_set_choice <= len(test_sets)):
        print("Invalid choice. Exiting evaluation.")
        return
    selected_test_set_path = test_sets[test_set_choice - 1][-1]

    # Load the selected test set
    test_set = QATestset.load(selected_test_set_path)

    # Step 4: Initialize Models and Build the RAG Chain
    print("\nInitializing models and RAG chain...")
    raw_data_path = "data/raw"
    processed_data_path = "data/processed/processed_data.pkl"
    folder_path = "eval_results"

    data = prepare_data(processed_data_path, raw_data_path)

    # Initialize models
    llm = init_llm(llm_name)
    embedding = init_emb(emb_name)

    # Setup vectorstore and retriever
    handler = VectorstoreHandler(vsts_dir, embedding, data)
    vectorstore = handler.init_vectorstore()
    retriever = handler.init_retriever(vectorstore, k=3)

    retriever = handler.init_retriever(vectorstore, k=3)
    retriever = None  # Initialize your retriever with the knowledge base
    chain = build_rag_chain(retriever, llm)
    answer_fn  = create_answer_fn(chain)

    # Step 5: Evaluate the RAG Chain
    print("\nEvaluating the RAG chain...")
    report = evaluate(
        answer_fn=answer_fn,
        testset=test_set,
    )
    report.save(folder_path)

    # Step 6: Display the Report
    display_evaluation_results(report)

