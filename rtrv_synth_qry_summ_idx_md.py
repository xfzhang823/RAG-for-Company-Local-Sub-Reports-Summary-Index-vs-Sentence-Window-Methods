"""
File: rtrv_synth_qry_summ_idx_md.py
Author: Xiao-Fei Zhang
Date: March 22, 2024

Description:
    This script is the 2nd half of the Summary Index RAG method.
    Specifically, it loads the index, retrieve, synthesize, and query

Usage:
    The script shoould be executed directly.

Dependencies:
    Requires multiple functions from llamaindex's library
"""

# Import libraries
from pathlib import Path
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    get_response_synthesizer,
    Settings,
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine

import openai
import os
from llama_index.llms.openai import OpenAI


# instantiate openai api and set llm

# LlamaIndex already put llm in some of their classes. LLM is often not specified, but it's there in the background.
# It's default to ChatGPT, unless set separately by the user.
# As the api key is stored in memory by LlamaIndex,
# depending on how you set up your openai api key, you may not need to enter the api key everytime.
openai.api_key = os.getenv("OPENAI_API_KEY")

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
# chatgpt = OpenAI(temperature=0, model="gpt-4")
Settings.llm = chatgpt


# Load index
# Path of the project directory
directory_path = r".\project directory"  # enter your working/project directory path here
index_directory = Path(directory_path) / "index_summ_idx_md" # enter the name of the directory where you saved the index to 

# Load index from disk
storage_context = StorageContext.from_defaults(persist_dir=index_directory)
doc_summary_index = load_index_from_storage(storage_context)


# Query (retrieve, synthesize, and query engine)

# Enter query
# List of example queries
QUERY_1 = "What is Deloitte's revenue in the US?"
QUERY_2 = "What is Deloitte's revenue in the UK and Canada?"
QUERY_3 = "What is Deloitte's revenue in Norway?"
QUERY_4 = "What is Deloitte's revenue per employee in Malaysia?"
QUERY_5 = "What is Deloitte's revenue per employee in Norway?"
QUERY_6 = "How much is Deloitte's revenue per employee in Norway?"
QUERY_7 = "What month does Deloitte's fiscal year end on?"
QUERY_8 = "What services does Deloitte provide?"

# Set which query to use
query = QUERY_8

# High-level querying (Note: this uses the default embedding-based form of retrieval)
query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
response = query_engine.query(query)

# print out result
print("High-Level Querying")
print(f"Query: {query}")
print(f"Answer: {response}\n\n")


# Embedding-based Retrieval (specify steps and custom parameters)
retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    similarity_top_k=3,  # default is 1 but set here to 3
)
retrieved_nodes = retriever.retrieve(query)
print("Embedding-based Retrieval")

# print out the node(s) retrieved
print(f"No. of nodes retrieved: {len(retrieved_nodes)}")
# print(f"Retrieved nodes: \n{retrieved_nodes[0].node.get_text()}\n")
print("Nodes retrieved:")
for i, node in enumerate(retrieved_nodes):
    print(f"Node {i+1}: \n{node.node.get_text()}\n")

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine with retriever and synthesizer
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query(query)

# print out query and response
print(f"Query: {query}")
print(f"Answer: {response}\n\n")


print("Done!")
