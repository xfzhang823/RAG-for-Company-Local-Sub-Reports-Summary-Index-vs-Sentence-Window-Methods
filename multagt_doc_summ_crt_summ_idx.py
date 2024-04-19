"""
File: multagt_doc_summ_crt_summ_idx.py
Author: Xiao-Fei Zhang
Date: Apr 17, 2024

Description:
    This module is the 2nd part of the multi-agent summarization RAG model, designed to set up a multi-document 
    agent system with summarization capabilities. It creates a summarization index by loading documents, inserting 
    metadata, and persisting each document using the DocumentSummaryIndex class.

Usage: This script should be executed directly.

Dependencies: Requires multiple functions from the llamaindex library.
"""

from pathlib import Path
import logging
import sys
import os

from llama_index.core import (
    DocumentSummaryIndex,
    SimpleDirectoryReader,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import openai

# Configure logging
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # Adjust logging as needed for detailed step-by-step tracing.

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Data directory path
data_directory_path = Path(
    r"C:\github\chatgpt\rag deloitte transparency reports\data\raw data"
)

# Mapping dict: keys => country, values => file names with PDF extension
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

# Setup sentence splitter and response synthesizer for document summarization
splitter = SentenceSplitter(chunk_size=1024)  # Node splitter for document parsing
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=False
)  # Synthesizer for creating summaries; async disabled to manage API rate limits

# LLM and embedding model configuration
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-ada-002", max_length=512
)

# Directory to persist the summarization index
persist_directory = Path(
    r"C:\github\chatgpt\rag deloitte transparency reports\index_multi_agent_summ"
)

# Create and persist a DocumentSummaryIndex for each set of documents
for country, docs in all_docs.items():
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    doc_summary_index.storage_context.persist(persist_directory / country)
