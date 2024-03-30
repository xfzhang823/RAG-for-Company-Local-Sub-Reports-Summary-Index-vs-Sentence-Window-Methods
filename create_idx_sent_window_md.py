"""
File: create_idx_sent_window_md.py
Author: Xiao-Fei Zhang
Date: March 22, 2024

Description:
    This script is the 1st part of the Sentence Window RAG method.
    Specifically, it reads multiple PDF documents (selected Deloitte's country level transparent report) into a document object.
    Indexes it into a vectorstore object: 
    - chunking the documents using the custom parser for sentence window to "small" chunks for searching 
    and "large" chunks to generate prompts for the LLM
    - using openai to embedd the chunks, and
    - create metadata.
    Saves the index to disk.functions to process and analyze customer feedback data.

Usage:
    The script shoould be executed directly.

Dependencies:
    Requires multiple functions from llamaindex's library
"""

# Import libs
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)  # just in case if you decide to use HF's embedding engine
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
import openai


# Setup
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
openai.api_key = os.environ["OPENAI_API_KEY"]
embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002", max_length=512)

# Embedding: you can comment out between HuggingFaceEmbedding and OpenAIEmbedding
Settings.llm = llm
Settings.embed_model = embed_model

# HuggingFace embedding
# embed_model = HuggingFaceEmbedding(
#     model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
# )

text_splitter = (
    SentenceSplitter()
)  # standard sentence splitter / split by sentence and each sentence is a chunk
Settings.text_splitter = text_splitter


# Load data, build, and index

# # Load data from docstore JSON file if we have already saved it
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

# Load original documents
data_directory_path = (
    r"\raw data"
) # enter your directory path of the document data to be read
documents = SimpleDirectoryReader(data_directory_path, filename_as_id=True).load_data()
# SimpleDirectoryReader method lets you read an entire directory or an individual files.
# You can use "input_files =" param to read just files.
print("document loaded.\n")


# Extract nodes

# Extract nodes w/t node sentence window parser
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(documents)
# window_size of 3 means that the parser will include:
# - 3 sentences before, the sentence itself, and 3 sentences after as the "window chunk",
# which will be used to generate the prompt into the LLM later
# It will also create 2 attributes in the metadata: window, original_text

# Extract nodes w/t just the standard sentence parser
base_nodes = text_splitter.get_nodes_from_documents(documents)

print("document parsed.\n")


# Build indexes and save/persist
# We are creating two indices:
# - base_index is w/o "windows"; it is used for comparison later on if you want to - it's optional
# - sent_index is w/t "windows"

# file paths for index storage
persist_path_base_index = r".\base_index" 
persist_path_sent_index = r".\sentence_index"
# enter your designated folders to save index files;
# if the immediate folders are not created, this method will create them for you;
# but root directory path must be correct.
# Indexing takes time and has LLM token charges,
# you need to make sure that code for persisting/saving index files to disk are correct
# to not waste time & money.


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
