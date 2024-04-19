"""
File: multagt_doc_summ_crt_vectstor_idx.py
Author: Xiao-Fei Zhang
Date: Apr 17, 2024

Description:
    This module is the 2nd part of the multi-agent summarization RAG model, which sets up a multi-document
    agent system. It creates a VectorStore index by loading documents into a nested dictionary, inserting 
    metadata (country), and persisting each to disk using the VectorStoreIndex class.

Usage: This script should be executed directly.

Dependencies: Requires multiple functions from the llamaindex library.
"""

import os
from pathlib import Path
import logging
import sys

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import openai

# Configure logging
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # Use logging only when detailed step tracking is necessary.

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Data directory path
data_directory_path = Path(
    r"C:\github\chatgpt\rag deloitte transparency reports\data\raw data"
)

# File name to country mapping
country_to_file_mapping = {
    "Australia": "deloitte-australia-2023-audit-transparency-report.pdf",
    "Canada": "deloitte-canada-2023-audit-transparency-report.pdf",
    "Denmark": "deloitte-denmark-2023-audit-transparency-report.pdf",
    "Malaysia": "deloitte-malaysia-2023-audit-transparency-report.pdf",
    "Norway": "deloitte-norway-2022-audit-transparency-report.pdf",
    "Slovakia": "deloitte-slovakia-2023-audit-transparency-report.pdf",
    "South Korea": "deloitte-south_korea-2023-audit-transparency-report.pdf",
    "UK": "deloitte-uk-2022-audit-transparency-report.pdf",
    "US": "deloitte-us-2023-audit-transparency-report.pdf",
}

# Read documents and insert country metadata
doc_set = {}
all_docs = {}
base_path = data_directory_path
for country, filename in country_to_file_mapping.items():
    file_path = base_path / filename
    country_docs = SimpleDirectoryReader(
        input_files=[file_path], filename_as_id=True
    ).load_data()
    for doc in country_docs:
        doc.metadata = {"country": country}

    if country not in all_docs:
        all_docs[country] = []
    all_docs[country].extend(country_docs)

# LLM settings
Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-ada-002", max_length=512
)

# Directory to persist index - vector store indices (not summarized)
persist_directory = Path(
    r"C:\github\chatgpt\rag deloitte transparency reports\index_multi_agent_vectorstore"
)

# Create and persist multiple VectorStore indices (one for each document/country)
for key, docs in all_docs.items():
    storage_context = StorageContext.from_defaults()
    vector_store_index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context
    )
    vector_store_index.storage_context.persist(persist_directory / key)
