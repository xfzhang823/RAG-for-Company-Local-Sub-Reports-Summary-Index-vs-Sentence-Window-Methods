# RAG for Company Centric Documents: Coding Walk-through

This repository contains the coding walk-through for the "Comparative Analysis of Summary Index and Node Sentence Window Methods in RAG for Local Subsidiary Documents" article. 
The code examples here are primarily based on the techniques discussed in the [LlamaIndex's Advanced RAG site](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/).

## Related Links
- [GitHub Repository]([https://github.com/xfzhang823/RAG-for-Company-Local-Sub-Reports-Summary-Index-vs-Sentence-Window-Methods/tree/main]): Access the script and data files related to this project.
- [Main Article](https://www.linkedin.com/pulse/comparative-analysis-summary-index-node-sentence-window-zhang-rhmse/?trackingId=q5IcvG8QRjil%2FS%2BZpc4lbQ%3D%3D): For an in-depth understanding of the goals, key concepts, and detailed output analysis of the RAG implementation.
- [Detailed Code Walkthrough](https://www.linkedin.com/pulse/comparison-document-summary-index-sentence-window-methods-zhang-bplae/): For explanations of each lines of code

## Repository Contents
The repository includes four application files for the RAG implementation and an optional index info viewer:
1. `create_idx_summ_idx_md.py`: Implements the Document Summary Index method for building the retrieval knowledgebase.
2. `rtrv_synth_qry_summ_idx_md.py`: Handles retrieval and querying using the Document Summary Index method.
3. `create_idx_sent_window_md.py`: Applies the Node Sentence Window method for constructing the retrieval knowledgebase.
4. `query_sent_window_md.py`: Manages retrieval and querying for the Node Sentence Window method.
5. `get_index_info.py`: A utility class to examine vectorstore index summaries.

## Implementation Notes
- Indexing processes, which take between 10 to 20 minutes due to their complexity and associated token costs, are designed to persist data to disk first. This approach facilitates separate execution for the retrieval and query operations, optimizing resource utilization and cost-efficiency.
- The code is structured to allow independent operation of indexing and querying, ensuring flexibility for future modifications or integration into larger projects.

## Getting Started
To use these scripts, clone the repository and install any necessary dependencies as outlined in the provided `requirements.txt` file. Ensure you have the appropriate environment set up for running Python scripts.

For a step-by-step guide on executing these scripts and a more detailed explanation of each file's role within the RAG framework, refer to the documentation provided within each script and the main article linked above.
