"""
File: multagt_doc_summ_crt_vectstor_idx.py
Author: Xiao-Fei Zhang
Date: Apr 17, 2024

Description:
    This module is the 3rd part of the multi-agent RAG model - the retrival/query process.

    It handles the setup and management of query engines for information retrieval from indices stored 
    in designated directories.

    It can manually swich between vector store (standard) and document summary indices for comparison, and 
    integrates with OpenAI's language models for response generation.

    The script is divided into the following parts:
    1. Initialization of directory paths and loading of environment configurations.
    2. Definition of utility functions to load indices from the storage and initialize query tools.
    3. Creation of query engines and tools, including sub-question engine/tool.
    4. Integration with OpenAI's LLMs for generating responses to user queries. 

    The system is designed to be modular, allowing for easy adaptation and expansion.

Usage:
    The script should be executed directly.

Dependencies:
    Requires multiple functions from the llamaindex library which provides tools for document indexing,
    query retrieval, and interfacing with OpenAI's language models. Specific dependencies include:
    - llamaindex.agent.openai
    - llamaindex.core
    - llamaindex.core.indices.document_summary
    - llamaindex.core.query_engine
    - llamaindex.llms.openai
    - openai

    External dependencies:
    - Python 3.8 or higher
    - openai library
    - pathlib, logging, os, sys standard libraries

"""

import os
import sys
from pathlib import Path
import logging

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    DocumentSummaryIndex,
    get_response_synthesizer,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

import openai

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directories for stored indices
vector_store_path = Path(
    r"...\index_multi_agent_vectorstore"
)    # input directory paths where indices are stored in
document_summary_path = Path(
    r"...\index_multi_agent_summ"
)    # input directory paths where indices are stored in


# Function to load indices from a given path
def load_indices(storage_path):
    """
    Load indices from a specified storage path.

    Parameters:
        storage_path (Path): The path to the directory containing index storage subdirectories.

    Returns:
        dict: A dictionary mapping subdirectory names to loaded index objects.
    """
    return {
        subdir.name.replace(" ", "_"): load_index_from_storage(
            StorageContext.from_defaults(persist_dir=subdir)
        )
        for subdir in storage_path.iterdir()
        if subdir.is_dir()
    }


# Load indices from storage
try:
    vector_store_indices = load_indices(vector_store_path)
    document_summary_indices = load_indices(document_summary_path)
    logging.info("Indices loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load indices: {e}")
    sys.exit(1)


# Initialize query tools
def create_query_tools(indices, description_template, response_mode=None):
    """
    Creates query tools for the indices.

    Parameters:
        indices (dict): A dictionary of indices to initialize query tools for.
        description_template (str): A template string for naming the metadata of the tools.
        response_mode (str, optional): The mode of response synthesizer to use. If not None,
                                       it initializes a RetrieverQueryEngine with the specified response mode.

    Returns:
        list: A list of QueryEngineTool objects initialized for each index.
    """
    query_tools = []
    for country, index in indices.items():
        if response_mode:
            retriever = DocumentSummaryIndexEmbeddingRetriever(
                index, similarity_top_k=2
            )
            response_synthesizer = get_response_synthesizer(response_mode=response_mode)
            query_engine = RetrieverQueryEngine(retriever, response_synthesizer)
        else:
            query_engine = index.as_query_engine()

        tool_metadata = ToolMetadata(
            name=f"{description_template}_{country}",
            description=f"useful for queries about the transparency report from Deloitte's subsidiary in {country}.",
        )
        query_tools.append(
            QueryEngineTool(query_engine=query_engine, metadata=tool_metadata)
        )
    return query_tools


# Create query engine tools
vector_store_query_tools = create_query_tools(vector_store_indices, "vecstor_idx")
document_summary_query_tools = create_query_tools(
    document_summary_indices, "doc_summ_idx", response_mode="tree_summarize"
)


# Create sub-question engines
def create_sub_question_tool(query_tools, description):
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_tools
    )
    return QueryEngineTool(
        query_engine=sub_question_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine", description=description
        ),
    )


vector_store_final_tools = vector_store_query_tools + [
    create_sub_question_tool(
        vector_store_query_tools,
        "Multiple local reports analysis using vector store indices",
    )
]
doc_summary_final_tools = document_summary_query_tools + [
    create_sub_question_tool(
        document_summary_query_tools,
        "Multiple local reports analysis using document summary indices",
    )
]


# Select tools based on user preference or configuration

# active_tools = (
#     vector_store_final_tools  # or doc_summary_final_tools based on configuration
# )

active_tools = (
    doc_summary_final_tools  # or vector_store_final_tools based on configuration
)

# Initialize OpenAI agent with selected tools
llm_4 = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4"))
llm_3 = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
agent = OpenAIAgent.from_tools(
    active_tools,
    llm=llm_3,
    system_prompt="You are an agent designed to answer queries about a set of local reports from Deloitte.",
    verbose=True,
)


# Example query and response

# Preset questions
QUERY_1 = "What is Deloitte's revenue in the US?"
QUERY_2 = "What is Deloitte's revenue in the UK and Canada?"
QUERY_3 = "What is Deloitte's revenue in Norway?"
QUERY_4 = "What is Deloitte's revenue per employee in Malaysia?"
QUERY_5 = "What is Deloitte's revenue per employee in Norway?"
QUERY_6 = "How much is Deloitte's revenue per employee in Norway?"
QUERY_7 = "What month does Deloitte's fiscal year end on?"
QUERY_8 = "What services does Deloitte provide?"
QUERY_9 = "What is Deloitte's revenue in different countries?"
QUERY_10 = "How many employees does Deloitte have in different countries"

query = QUERY_10
response = agent.chat(query)
print(f"Agent: {response}")
