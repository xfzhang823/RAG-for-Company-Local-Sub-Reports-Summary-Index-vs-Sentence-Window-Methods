"""
File: query_sent_window_md.py
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
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    # get_response_synthesizer,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from pathlib import Path
import os
import openai


# instantiate openai api and set llm
openai.api_key = os.environ[
    "OPENAI_API_KEY"
]  # if have you have set up your api key in your environment
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = OpenAIEmbedding(
    model_name="text-embedding-ada-002", max_length=512
)  # I think openai do not allow you to use gpt-3.5 for embedding anymore;
# therefore, you have to use its text embedding model/engine

Settings.llm = llm
Settings.embed_model = embed_model


# Load index from disk
# Path of the project directory
directory_path = r"C:\github\chatgpt\rag deloitte transparency reports"  # working directory of your project
sent_index_directory = (
    Path(directory_path) / "index_sent_window_md/sentence_index"
)  # sub-folers where you keep the sentence index

# Load index from disk
storage_context = StorageContext.from_defaults(persist_dir=sent_index_directory)
sentence_index = load_index_from_storage(storage_context)


# Query

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

# Query
query_engine = sentence_index.as_query_engine(
    similarity_top_k=3,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
sent_window_response = query_engine.query(query)


# Print out results
print("Metadata Replacement + Node Sentence Window")

# print out the original text of the node(s) retrieved
print(f"No. of nodes retrieved: {len(sent_window_response.source_nodes)}")
print("Nodes retrieved:")
for i, node in enumerate(sent_window_response.source_nodes):
    sent = node.node.metadata["original_text"]
    window = node.node.metadata["window"]
    print(f"Node {i+1}: original text\n{sent}")
    print("-----------------------------")
    print(f"Node {i+1}: window\n{window}")

# The query engine extracts the "original_text" and "window" from the source nodes in the index.
# You can find them in "metadata_dict" within the "default_vector_store.json" file you persisted to disk.
# Both are metadata attributes created when the sentence_index is built,
# b/c we used SentenceWindowNodeParser to to chunk the documents:
#  - the parser creates a 'window' of text around each sentence in the document,
# & this window of text is stored in the node's metadata under the key 'window'.
# - it also created the original_text metadata attribute, which contains
# the original sentence that the window was created around.

# print out the response (answer) to the query
print("Query Result")
print(f"Query: {query}")
print(f"Answer: {sent_window_response}\n")

print("Done!")
