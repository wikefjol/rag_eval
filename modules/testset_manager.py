import os
from giskard.rag import QATestset
from modules.utils import compute_dataset_hash, generate_test_set_path

def save_test_set(test_set, model_name, embedding_name, dataset, base_dir="data/test_sets"):
    """
    Save the test set in a structured directory.

    Args:
        test_set (QATestset): The test set to save.
        model_name (str): The model name (e.g., "ChatGPT3.5-turbo").
        embedding_name (str): The embedding name (e.g., "hf-mpnet-base-v2").
        dataset (pd.DataFrame): The dataset used to generate the test set.
        base_dir (str): The base directory for saving test sets.
    """
    dataset_hash = compute_dataset_hash(dataset)
    file_path = generate_test_set_path(base_dir, model_name, embedding_name, dataset_hash)
    test_set.save(file_path)
    print(f"Test set saved at: {file_path}")

def list_test_sets(base_dir="data/test_sets"):
    """
    List all available test sets in the structured directory.

    Args:
        base_dir (str): The base directory for test sets.

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

def choose_test_set(base_dir="data/test_sets"):
    """
    Prompt the user to choose an available test set.

    Returns:
        QATestset: The loaded test set or None if no valid selection is made.
    """
    test_sets = list_test_sets(base_dir)
    if not test_sets:
        print("No test sets available.")
        return None

    print("Available Test Sets:")
    for idx, (model, embedding, dataset_hash, file_path) in enumerate(test_sets, start=1):
        print(f"{idx}. Model: {model}, Embedding: {embedding}, Dataset Hash: {dataset_hash}, File: {os.path.basename(file_path)}")

    try:
        choice = int(input("Enter the number of the test set to use: "))
        if 1 <= choice <= len(test_sets):
            chosen_file = test_sets[choice - 1][-1]
            return QATestset.load(chosen_file)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")
    return None
