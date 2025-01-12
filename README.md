

# Retrieval-Augmented Generation (RAG) Evaluation Project

This project is a comprehensive framework designed to evaluate RAG models. It offers tools for generating test sets, running RAG pipelines, and evaluating models with detailed reports on performance and weaknesses. The project is a work in progress, with a focus on building robust tools and interfaces for RAG evaluation.

---
## Aim
Model agnostic production RAG: To be able to freely set up rags with any combination of llms & embedding models - let the users select which model/embeddings they use and give feedback for which work best?
    - Key feature: Creating vectorstores takes time. So we want to store and hash vectorstores based on llm, embedding model and dataset used to create it, so if that setup occurrs again, we don't have to recreate the vectorstore - this bug took ages to track down
RAG Evaluation a) : To be able to use "heavier" or more expensive llms to produce testsets that the prodction rags can be evaluated against - Seems to work, but generating the test sets takes time and I haven't increased model complexity yet
RAG Evaluation b) : To be able to input expert QA pairs as json and create a testset from that, against which the production rags can be evaluated against - To be done.  
RAG Evaluation c) : Combination of a and b, either/or both by i) Pretraining an llm based on the QA's, ii) augment and weigh the questions in the test set, so that expert answers weigh more heavily than simulated answers. - To be done
    - Key feature: giskard: opensource test set generation, different types of quesionts for different parts of the rag , see https://docs.giskard.ai/en/stable/reference/rag-toolset/testset_generation.html
    - Key feature: Model agnosticism. Wrappers for the llm and embdding models so all of them work with the giskard suite used for evaluation - this bug also took ages to track down
    - Key feature: Save and load testsets that are "approved", or checked. - To be done, need to decouple some things for this to work
    - Key feature: After having finetuned llm and generated augmented testset, send out simulated 10 answers for each question (maybe 2 questions (20 answers) per day for 2-4 weeks), and let experts grade each answer => larger expert QA pairs which we then can iterate and iterate and iterate, hopefully converging towards something very good. 
        

## Current Status

- **Notebook Access Point:** The best way to interact with this project right now is through the Jupyter notebook. It provides a step-by-step guide to using the features.

### Known Issues
0. Everything is quite messy still. Long way to go. 
1. Started working on the terminal interface - had a breakdown - left it for later
2. Error handling in edge cases is limited.
3. Evaluation logic needs to be slightly better. 
4. Need to fix so that I can load test sets and use in the eval

#### Notes about modules: 
- eval_tools has a long way to go
- giskard_wrappers is not too shabby
- knowledge_base is not too shabby, might need a handler later
- models work quite nicely - i might need to find out if I can get model information from model objects (llm/embedding), or how to handle names better for hashing purposes
- preprocess works okay -  havent been there for a long time
- rag_interface - i have nothing to say in my defense. It is a mess and should be forgotten about for now, started to work on the interface too early. But as is it is not used either. 
- RAG_tools needs to be renamed, and the create_answer_fn should probably be moved to the giskard_wrappers module. Also, by looking in the testsets it looks like some of the rag answers are not parsed correctly. 
- testset_manager has a long way to go
- utils is just a "others" at this moment. 
- tbh the entire codebase needs to be refactored, but has not been priority. first to get each piece to work somwhat robustly

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