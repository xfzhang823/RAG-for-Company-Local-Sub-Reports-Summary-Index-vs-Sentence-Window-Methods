"""
File: create_idx_sent_window_md.py
Author: Xiao-Fei Zhang
Date: March 22, 2024

Description:
    This script is the 1st part of the Sentence Window RAG method.
    Specifically, it reads multiple PDF documents (selected Deloitte's country level transparent report) into a document object.
    Indexes it into a vectorstore object: 
    - chunking the documents, 
    - using openai to embedd the chunks, and
    - create metadata.
    Saves the index to disk.functions to process and analyze customer feedback data.

Usage:
    The script shoould be executed directly.

Dependencies:
    Requires multiple functions from llamaindex's library
"""

# Import libs
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import Settings
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex

import os
import openai


# Setup
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

text_splitter = SentenceSplitter()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
openai.api_key = os.environ["OPENAI_API_KEY"]
embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002", max_length=512)


# Embedding: comment out between HuggingFaceEmbedding and OpenAIEmbedding
# HuggingFace embedding
# embed_model = HuggingFaceEmbedding(
#     model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
# )


Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter


# Load data, build, and index
from llama_index.core import SimpleDirectoryReader


# # We can load the docstore from the JSON file if we have already saved it
# import json
# from llama_index.core import Document

# doc_store_f_path = (
#     r"C:\github\chatgpt\rag deloitte transparency reports\docstore\doc_store.json"
# )
# with open(doc_store_f_path, "r") as f:
#     documents_dict = json.load(f)
# # now convert dictionaries back to Document objects
# doc_store = [Document.from_dict(doc_dict) for doc_dict in documents_dict]
# documents = SimpleDirectoryReader(
#     input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
# ).load

# Or just read the data again (which does not take that long)

# Load data
data_directory_path = (
    r"C:\github\chatgpt\rag deloitte transparency reports\data\raw data"
)
documents = SimpleDirectoryReader(data_directory_path, filename_as_id=True).load_data()
print("document loaded.\n")

# Extract nodes:
# Nodes w/t sentence window parser
nodes = node_parser.get_nodes_from_documents(documents)

# Base nodes: extracted using the standard sentence parser
base_nodes = text_splitter.get_nodes_from_documents(documents)
print("document parsed.\n")


# Build indexes and save/persist (b/c this takes long time, you want to index & persist right away)

# file paths for index storage
# (this part has to be absolutely correct b/c the indexing is long and expensive;
# if "saving goes wrong", you have to redo the entire indexing!)
persist_path_base_index = r"C:\github\chatgpt\rag deloitte transparency reports\index_sent_window_md\base_index"
persist_path_sent_index = r"C:\github\chatgpt\rag deloitte transparency reports\index_sent_window_md\sentence_index"
# if base_index or sentence_index (immediate folders are not created, llamaindex will create them for you;
# however, the root directory path must be correct)

# index and store sentence index (w/t windows)
sentence_index = VectorStoreIndex(nodes)
print("sentence indexing done.")
sentence_index.storage_context.persist(persist_dir=persist_path_sent_index)
print("sentence index saved.")

# index and store base index
base_index = VectorStoreIndex(base_nodes)
print("base indexing done.")
base_index.storage_context.persist(persist_dir=persist_path_base_index)
print("base index saved.")
print("All DONE!")
