

# Retrieval-Augmented Generation (RAG) Evaluation Project

## Intended use case:
User will provide documents they need help to interpret. They can select an llm and an embedding model to create a rag. Hopefully they can mark answers and thus produce training data. 
 
---
## Aim

We want users to have the best RAG possible. To achieve this, the project aims to provide a structured, reproducible framework to evaluate RAG performance. This project leverages [Giskard](https://docs.giskard.ai) as a key tool in building and assessing RAG systems.

---

## Challenges

### Vectorstore Management  
Vectorstore creation is time-intensive. This project includes a **Vectorstore Handler** to store and hash vectorstores based on the LLM, embedding model, and dataset. If the same configuration is used again, the handler retrieves the existing vectorstore instead of recreating it (this bug took a long time to resolve).

### Model Agnosticism  
Switching between RAGs built from different LLMs and embedding models is not trivial. In particular the vectorstore needs to match, otherwise we get dimension issues (after having waited for the code to compile for 5 minutes, mind you). Further, giskards generate_tests expects a specific interface which openai models don't match. So after the vectorstore issue was solved wrappers for llm and embedding models were built to ensure the framework is model-agnostic. Currently tested on:
- **LLMs**: OpenAI and Ollama models.
- **Embedding Models**: OpenAI and HuggingFace models.

### Test Set Challenges  
- **Slow Generation**: Test set generation using an LLM provides rich metadata (e.g., ground truth, reference context) but is very slow. A mechanism is needed to save and load test sets without requiring the corresponding vectorstore (or at least would be nice, might be possible to load vectorstore as well - need to lookup why the testset needs the knowledgebase).
- **QA JSON Input**: Generating test sets from QA JSONs is theoretically straightforward but lacks the detailed metadata. Relying on experts for metadata creation may not be feasible. Might need to include this in another stage of the process such as llm finetuning. 

---

## Dream Goals

### Model-Agnostic Production RAG and user feedback  
Enable users to set up RAGs with any combination of LLMs and embedding models and receive feedback on the configurations that work best.  
- **Key Feature**: Hashing and storing vectorstores by LLM, embedding model, and dataset ensures no redundant vectorstore creation.
- **Key Feature**: Managing and exploiting user feedback

### RAG Evaluation  
Use one LLM-embedding combination for the production RAG and another for evaluation.

#### Key Ideas:
- **Heavier LLMs for Evaluation**: Use complex, closely monitored, and finetuned LLMs to generate test sets and evaluate RAGs.  
  - Status: Initial implementation works but needs refinement. Model complexity has not yet been increased due to time concerns.
- **QA JSON Inputs**: Create test sets from expert QA pairs as JSON files.  
  - Status: TODO - Consider pretraining LLMs on QA pairs to mimic expert evaluation.
- **Hybrid Approach**: Combine LLM-generated and expert-augmented test sets:
  1. Pretrain LLMs with QA pairs. 
  - Status: TODO - feasible but simulated context from the rag remains an issue
  2. Weight questions to prioritize expert answers over llm-generated answers.
  - Status: TODO

#### Key Features:
- **Giskard Integration**: Open-source test set generation targets specific RAG weaknesses. [Giskard Documentation](https://docs.giskard.ai/en/stable/reference/rag-toolset/testset_generation.html).  
- **Model-Agnostic Wrappers**: Ensure all LLMs and embeddings work seamlessly with Giskard.  
- **Test Set Management**: Save and load "approved" or verified test sets to continually and quickly evaluate rags. 
- **Iterative Expert QA Refinement**: Use expert feedback on generated answers to iteratively refine QA pairs, incorporating this into the test sets.

---

## Current Status

### Notebook Access Point  
The Jupyter notebook provides a simple pass through the basic functionalities of the project. Loads of work has been done to prepare for more complex setups, but: time. 

### Known Issues
- Preprocessing or retriever fragmentation: Output like "Wha  Does Ho izo  Eu ope Look Like?" suggests tokenization issues. Check `./eval_results/*/knowledge_base.jsonl`.  
- Terminal interface development stalled and needs a complete overhaul - notebooks it is... for a while.  
- Limited error handling and logging.  
- Need to improve evaluation workflows and how reports from different setups are compared. The evaluation function itself is quite simple, but need a framework for saving and comparing evaluation reports. Might be time for config-file management. 
- Test set loading feasibility remains uncertain; decoupling test sets from vectorstores is an open question.

---

## Notes About Modules

- **`eval_tools`**: Needs significant development.  
- **`giskard_wrappers`**: Stable and functional.  
- **`knowledge_base`**: Reliable but may need a handler in the future.  
- **`models`**: Works well but requires better handling of model metadata for hashing.  
- **`preprocess`**: Functional but outdated.  
- **`rag_interface`**: A messy prototype that requires a major overhaul. Currently unused.  
- **`RAG_tools`**: Needs renaming. The `create_answer_fn` function should move to `giskard_wrappers`. Some RAG answers are not parsed correctly in test sets.  
- **`testset_manager`**: Needs substantial work.  
- **`utils`**: A catch-all module that requires reorganization.  
- **Codebase**: Refactoring is needed but not yet prioritized.

---

### Notes About Tests
- Only one test exists: A minimal check pairing all embedding models and LLMs, constructing a chain, and asking it to tell a joke. It ensures no incompatible configurations. Runs with pytest and saves a log 

## Directory Structure

```
.
├── data/                # Processed, raw, and test set data
│   ├── processed/       # Processed datasets
│   ├── raw/             # Raw input datasets (equivalent to what merima called mc-data)
│   ├── test_sets/       # Structured test sets
│   └── vectorstores/    # Persistent vector stores
├── eval_results/        # Evaluation reports and results
├── modules/             # Core Python modules
├── notebooks/           # Jupyter notebooks for interactive use
├── tests/               # Unit tests for modules
├── venv/                # Virtual environment
├── requirements.txt     # Python dependencies
└── main.py              # Terminal interface (WIP)
```

---

## Usage
use the notebook for now, and don't look under the hood 
---

## Configuration
The project supports various LLMs and embedding models. You can configure these in the terminal interface or notebook:
- **LLMs:** Available options include `ChatGPT3.5-turbo`, `Llama3.2-3b`, etc.
- **Embeddings:** Includes Hugging Face models like `sentence-transformers/all-mpnet-base-v2` and OpenAI embeddings.
Adding new options should be relatively straightforward, but not a priority. 

---




---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature name"
   ```
4. Push your branch and create a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).