import os
import json
import hashlib
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from modules.utils import compute_dataset_hash
from tqdm import tqdm


class VectorstoreHandler:
    """
    A handler for managing Chroma vectorstores, ensuring compatibility with embeddings
    and datasets, and providing an intuitive API for initialization and retrieval.
    """
    def __init__(self, persist_directory, embedding, dataset, force_rebuild=False):
        """
        Initialize the VectorstoreHandler.

        Parameters:
            persist_directory (str): Base directory for storing vectorstores.
            embedding: The embedding model to use.
            dataset (pd.DataFrame): Dataset containing text data.
            force_rebuild (bool): Whether to rebuild the vectorstore even if it exists.
        """
        self.persist_directory = persist_directory
        self.embedding = embedding
        self.dataset = dataset
        self.force_rebuild = force_rebuild
        self.dataset_hash = self._compute_dataset_hash(dataset)
        self.embedding_name = getattr(embedding, "model_name", "unknown_embedding").replace("/", "_").lower()
        self.vectorstore_dir = os.path.join(self.persist_directory, self.embedding_name, self.dataset_hash)

    def _compute_dataset_hash(self, data):
        """
        Compute a unique hash for the dataset by hashing the concatenated text.

        Parameters:
            dataset (pd.DataFrame): Dataset containing text data.

        Returns:
            str: SHA-256 hash of the dataset.
        """
        return compute_dataset_hash(data)

    def _load_metadata(self):
        """
        Load metadata from the vectorstore directory.

        Returns:
            dict: Metadata dictionary, or None if no metadata file exists.
        """
        metadata_path = os.path.join(self.vectorstore_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def _save_metadata(self, metadata):
        """
        Save metadata to the vectorstore directory.

        Parameters:
            metadata (dict): Metadata dictionary to save.
        """
        metadata_path = os.path.join(self.vectorstore_dir, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _validate_metadata(self, metadata):
        """
        Validate that the metadata matches the current embedding and dataset hash.

        Parameters:
            metadata (dict): Metadata dictionary to validate.

        Returns:
            bool: True if the metadata is valid, False otherwise.
        """
        expected_metadata = {
            "embedding_name": self.embedding_name,
            "dataset_hash": self.dataset_hash,
        }
        return metadata == expected_metadata

    def init_vectorstore(self):
        """
        Initialize or load the vectorstore, validating metadata and rebuilding if necessary.

        Returns:
            Chroma: The initialized or loaded Chroma vectorstore.
        """
        # Check if vectorstore exists and matches current setup
        existing_metadata = self._load_metadata()
        if existing_metadata and self._validate_metadata(existing_metadata) and not self.force_rebuild:
            print(f"Loading existing vectorstore from {self.vectorstore_dir}...")
            return Chroma(embedding_function=self.embedding, persist_directory=self.vectorstore_dir)

        if self.force_rebuild and existing_metadata:
            print(f"Force rebuilding vectorstore: deleting {self.vectorstore_dir}...")
            self._delete_vectorstore()

        # Create a new vectorstore
        print("Creating a new vectorstore...")
        chunks = self._create_chunks()

        vec_store = Chroma(embedding_function=self.embedding, persist_directory=self.vectorstore_dir)
        print("Adding documents to the vectorstore...")
        vec_store.add_documents(chunks)

        # Save updated metadata
        self._save_metadata({
            "embedding_name": self.embedding_name,
            "dataset_hash": self.dataset_hash,
        })
        print(f"Vectorstore initialized and metadata saved successfully in {self.vectorstore_dir}.")
        return vec_store

    def _create_chunks(self):
        """
        Split the dataset into chunks for the vectorstore.

        Returns:
            list: List of text chunks.
        """
        chunks = []
        sem_text_splitter = SemanticChunker(self.embedding)
        print("Splitting text into chunks...")
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc="Processing documents"):
            text_chunks = sem_text_splitter.create_documents([row["text"]])
            chunks.extend(text_chunks)
        return chunks

    def _delete_vectorstore(self):
        """
        Delete the vectorstore directory and its contents.
        """
        if os.path.exists(self.vectorstore_dir):
            for root, dirs, files in os.walk(self.vectorstore_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.vectorstore_dir)

    def init_retriever(self, vectorstore, k):
        """
        Initialize a retriever from the vectorstore.

        Parameters:
            vectorstore (Chroma): The vectorstore to use.
            k (int): Number of documents to retrieve.

        Returns:
            retriever: A retriever instance.
        """
        return vectorstore.as_retriever(search_kwargs={"k": k})
