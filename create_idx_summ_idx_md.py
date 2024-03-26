"""
File: create_idx_summ_idx_md.py
Author: Xiao-Fei Zhang
Date: March 22, 2024

Description:
    This script is the 1st part of the Summary Index RAG method.
    Specifically, it reads multiple PDF documents (selected Deloitte's country level transparent report) into a document object.
    Indexes it into a vectorstore object: 
    - chunking the documents, 
    - using openai to embedd the chunks, and
    - create hiearchical summary metadata.
    Saves the index to disk.functions to process and analyze customer feedback data.

Usage:
    The script shoould be executed directly.

Dependencies:
    Requires multiple functions from llamaindex's library
"""

# Import libs
import os
import openai

from llama_index.core import (
    DocumentSummaryIndex,
    SimpleDirectoryReader,
)
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext
from llama_index.core.node_parser import SentenceSplitter


# Read documents
data_directory_path = (
    r"\data\raw data"     # enter the location of your data file folder here
)
reader = SimpleDirectoryReader(input_dir=data_directory_path, filename_as_id=True)
doc_store = reader.load_data()


# Build the index
openai.api_key = os.environ["OPENAI_API_KEY"]
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")

# define ServiceContext with chunk_size_limit
service_context = ServiceContext.from_defaults(
    llm=chatgpt, chunk_size_limit=1024
)  # default limit is 512 usually but set to higher for faster speed
# ServiceContext is deprecated, but still works

splitter = SentenceSplitter(chunk_size=1024)

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=False
)  # use_async is supposed to be True for faster processing, but set to False here b/c the openai api is hitting the rate limit

doc_summary_index = DocumentSummaryIndex.from_documents(
    doc_store,
    service_context=service_context,
    # use service_context to specify embedding engine to use and control token rate to avoid hitting the limit;
    # you can also use Settings now
    # llm=chatgpt,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
)
# this part take a long time
"""
    Note:
    The response_synthesizer should be a process AFTER embedding/indexing. 
    But here it is not used to generate responses to queries.
    The response_synthesizer is used in the indexing/embedding stage to generate summaries for the chunks.
    However, the mode is set to "tree_summarize". The summaries are hierarchical:
    - The process starts by summarizing individual chunks of the document. 
    - Then initial summaries are grouped together and summarized again, creating a second level of summary.
    - It repeats, w/t each round of summarization aggregating and summarizing again and again, 
    until a single summary for all the chunks is produced.
    The summaries are then stored in the index along with the embeddings, as part of the metadata.
    They are then used during the retrieval and synthesizing stages.
"""

# Store index using llama-index's own storage method
persist_path = r"\index"     # input your directory folder location here
doc_summary_index.storage_context.persist(persist_path)
