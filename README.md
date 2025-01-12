

# Retrieval-Augmented Generation (RAG) Evaluation Project

---
## Aim

### Model-Agnostic Production RAG
To freely set up RAGs with any combination of LLMs and embedding models, allowing users to select which model/embeddings they use and receive feedback on which configurations work best.
- **Key Feature**: Creating vectorstores takes time. We aim to store and hash vectorstores based on the LLM, embedding model, and dataset used to create them. If the same setup occurs again, the vectorstore won't need to be recreated (this bug took ages to track down).

### RAG Evaluation
**a)** Use "heavier" or more expensive LLMs to produce test sets that production RAGs can be evaluated against.  
- Status: Seems to work, but generating test sets takes time, and increasing model complexity hasn't been tested yet.

**b)** Input expert QA pairs as JSON to create test sets against which production RAGs can be evaluated.  
- Status: To be done.

**c)** Combine (a) and (b) by:
  1. Pretraining an LLM based on the QA pairs.
  2. Augmenting and weighting the questions in the test set, ensuring expert answers weigh more heavily than simulated ones.  
- Status: To be done.

#### Key Features:
- **Giskard Integration**: Use open-source test set generation with different types of questions targeting specific parts of the RAG. See [Giskard Documentation](https://docs.giskard.ai/en/stable/reference/rag-toolset/testset_generation.html).
- **Model-Agnostic Wrappers**: Wrappers for LLMs and embedding models to ensure compatibility with the Giskard evaluation suite (another bug that took ages to track down).
- **Test Set Handling**: Save and load test sets that are "approved" or verified.  
  - Status: To be done (requires decoupling for better functionality).
- **Iterative Improvement**: After fine-tuning the LLM and generating an augmented test set, simulate 10 answers per question (e.g., 2 questions per day for 2–4 weeks). Let experts grade these answers to iteratively expand and refine the QA pairs, converging toward a robust system.

---

## Current Status

### Notebook Access Point
The best way to interact with this project right now is through the Jupyter notebook, which provides a step-by-step guide for exploring the project's features.

### Known Issues
1. The project is still messy and incomplete. Long way to go.
2. Terminal interface development stalled and needs a complete revisit.
3. Error handling in edge cases is limited.
4. Evaluation logic requires refinement.
5. Test set loading and evaluation need improvement.

---

## Notes About Modules

- **`eval_tools`**: Still underdeveloped; significant work remains.  
- **`giskard_wrappers`**: Functional and in good shape.  
- **`knowledge_base`**: Works well but might need a handler in the future.  
- **`models`**: Functions nicely. Need better handling of model names or metadata for hashing purposes.  
- **`preprocess`**: Functional but hasn't been updated in a while.  
- **`rag_interface`**: A complete mess. Started working on the interface too early. Currently unused but needs a major overhaul.  
- **`RAG_tools`**: Needs renaming. The `create_answer_fn` function should be moved to the `giskard_wrappers` module. Some test set RAG answers appear to be incorrectly parsed.  
- **`testset_manager`**: Far from complete.  
- **`utils`**: Currently an "other" category; needs organization.  
- **Overall Codebase**: Refactoring is overdue but hasn't been prioritized. The focus remains on making individual components robust first.
### Notes about tests: 
- Only wrote a one test so far: Pairs all possible combinations of embedding models and llms and constructs a small chain of model|parser and asks the chain to tell us a joke - just to see that there are no incompatible options. 
--- 

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