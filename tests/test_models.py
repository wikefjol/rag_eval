import sys
import os
import pytest
import logging
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import models

# Define mock LLMs and Embeddings
available_llms = models.AVAILABLE_LLMS
available_embs = models.AVAILABLE_EMBS
TEST_PROMPT = "tell me a joke"

# Ensure the logs directory exists
log_dir = os.path.dirname(__file__)  # Directory of the test script
log_file = os.path.join(log_dir, "test_results.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="w"  # Overwrite the log file on each test run
)


@pytest.mark.parametrize("llm_name", available_llms.keys())
@pytest.mark.parametrize("emb_name", available_embs.keys())
def test_llm_embedding_combinations(llm_name, emb_name):
    """
    Test all combinations of LLMs and embeddings to ensure they work together.
    """
    logging.info(f"Starting test for LLM={llm_name}, Embedding={emb_name}")
    
    # Initialize LLM
    try:
        llm = models.init_llm(llm_name)
    except Exception as e:
        pytest.fail(f"Failed to initialize LLM: {llm_name}. Error: {e}")
    
    # Initialize embedding
    try:
        embedding = models.init_emb(emb_name)
    except Exception as e:
        pytest.fail(f"Failed to initialize embedding: {emb_name}. Error: {e}")
        
    # Test chain invocation
    try:
        parser = StrOutputParser()
        chain = llm | parser | (lambda output: output.replace("\n", " ").strip())
        ans = chain.invoke(TEST_PROMPT)

        # Log the response
        if ans:
            logging.info(f"LLM={llm_name}, Embedding={emb_name} responded: {ans[:50]}...")  # Log first 50 characters
        else:
            pytest.fail(f"No response for LLM={llm_name}, Embedding={emb_name}")

        # Assertions
        assert ans is not None, f"No response for LLM={llm_name}, Embedding={emb_name}"
        assert len(ans.strip()) > 0, f"Empty response for LLM={llm_name}, Embedding={emb_name}"
    except Exception as e:
        pytest.fail(f"Failed to invoke chain for LLM={llm_name}, Embedding={emb_name}. Error: {e}")
