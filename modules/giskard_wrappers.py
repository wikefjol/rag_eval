class GiskardEmbeddingAdapter:
    """
    Adapter for LangChain embedding models to match Giskard's expected interface.
    """

    def __init__(self, embedding_model):
        """
        Initialize the adapter.

        Args:
            embedding_model: A LangChain embedding model (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings).
        """
        self.embedding_model = embedding_model

    def embed(self, texts):
        """
        Embed a list of texts to match Giskard's `embed` method interface.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        if not isinstance(texts, list):
            raise ValueError("Input to `embed` must be a list of strings.")
        try:
            # Use LangChain's embedding method for documents
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Error embedding texts with {type(self.embedding_model).__name__}: {e}")


class GiskardLLMAdapter:
    """
    Adapter for LangChain LLMs to match Giskard's expected interface.
    """

    def __init__(self, llm):
        """
        Initialize the adapter.

        Args:
            llm: A LangChain LLM (e.g., ChatOpenAI, OllamaLLM).
        """
        self.llm = llm

    def complete(self, messages, temperature=0.0):
        """
        Generate a completion for a list of messages, matching Giskard's `complete` method.

        Args:
            messages (List[ChatMessage]): List of chat messages (each with `role` and `content`).
            temperature (float): Sampling temperature for generation.

        Returns:
            Response: A mock response object with a `content` attribute containing the generated text.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Messages must be a non-empty list of ChatMessage objects.")

        try:
            # Extract the content of the first message (assuming single-message input)
            prompt = messages[0].content

            # Call LangChain's LLM invoke method
            response = self.llm.invoke(prompt)

            # Create a response object with text extraction
            return type("Response", (), {
                "content": str(response),
                "text": str(response),
                "strip": lambda x=None: str(response).strip()
            })()
        except Exception as e:
            raise RuntimeError(f"Error completing messages with {type(self.llm).__name__}: {e}")