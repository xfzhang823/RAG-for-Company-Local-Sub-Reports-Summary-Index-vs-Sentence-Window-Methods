# About the File:
# This app reads selected Deloitte's country level transparent report docs (pdf format) into a
# document object.
# Indexes it into a vectorstore object, including
# chunking the documents, using openai to embedd and create hiearchical summary metadata.
# Then saves the index to disk.


# Import libs
from llama_index.core import Document
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core import DocumentSummaryIndex
from llama_index.core import ServiceContext
from llama_index.core.node_parser import SentenceSplitter

import openai
import os
import json
from pathlib import Path

# Load doc_store from disk

# open json file
doc_store_f_path = (
    r"C:\github\chatgpt\rag deloitte transparency reports\docstore\doc_store.json"
)
with open(doc_store_f_path, "r") as f:
    documents_dict = json.load(f)

# Convert dictionaries back to Document objects
doc_store = [Document.from_dict(doc_dict) for doc_dict in documents_dict]


# Build the index
openai.api_key = os.environ["OPENAI_API_KEY"]
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")

# define ServiceContext with chunk_size_limit
service_context = ServiceContext.from_defaults(
    llm=chatgpt, chunk_size_limit=1024
)  # default limit is 512, usually
# ServiceContext is deprecated, but still works for now

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
    Important Note:
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

# Store index using llama-index's own storage methods
persist_path = r"C:\github\chatgpt\rag deloitte transparency reports\index"
doc_summary_index.storage_context.persist(persist_path)
