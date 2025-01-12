from giskard.rag import KnowledgeBase, generate_testset as giskard_generate_testset, evaluate as gk_eval


def create_knowledge_base(data, columns, seed=42, min_topic_size=2, chunk_size=2048):
    """
    Creates a knowledge base from the provided data.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the knowledge base data.
        columns (list): List of columns to use for the knowledge base.
        seed (int): Random seed for reproducibility.
        min_topic_size (int): Minimum number of documents to form a topic.
        chunk_size (int): Number of documents to embed in a single batch.

    Returns:
        KnowledgeBase: The created knowledge base object.
    """
    return KnowledgeBase(
        data=data,
        columns=columns,
        seed=seed,
        min_topic_size=min_topic_size,
        chunk_size=chunk_size,
    )


def create_testset(knowledge_base, num_questions=10, language="en", agent_description="This LLM assists with queries using a knowledge base."):
    """
    Generates a test set from the knowledge base.

    Parameters:
        knowledge_base (KnowledgeBase): The knowledge base to generate questions from.
        num_questions (int): Number of questions to generate.
        language (str): Language of the questions (default is English).
        agent_description (str): Description of the agent for generating relevant questions.

    Returns:
        QATestset: The generated test set.
    """
    testset = giskard_generate_testset(
        knowledge_base=knowledge_base,
        num_questions=num_questions,
        language=language,
        agent_description=agent_description,
    )
    return testset


def evaluate(rag_chain, testset, knowledge_base):
    """
    Evaluates the RAG chain using the test set and knowledge base.

    Parameters:
        rag_chain: The RAG chain for generating answers.
        testset: The test set to evaluate against.
        knowledge_base (KnowledgeBase): The knowledge base used for the evaluation.

    Returns:
        dict: Evaluation report.
    """
    return gk_eval(rag_chain, testset=testset, knowledge_base=knowledge_base)
