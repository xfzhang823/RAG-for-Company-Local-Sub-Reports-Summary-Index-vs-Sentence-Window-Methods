"""
File: vector_store_summary.py
Author: Xiao-Fei Zhang
Date: April 1, 2024

Description:
    This script prints metrics of an index and a summary of the metrics.
    Also save the summary to a text file & node ids to an excel file.

Usage:
    The script shoould be executed directly.

Dependencies:
    Requires multiple functions from llamaindex's library and pandas.
"""

# Import libraries
from pathlib import Path
import os
import pandas as pd

# import numpy as np

from llama_index.core import StorageContext, load_index_from_storage

import spacy

nlp = spacy.load("en_core_web_sm")  # spaCy for tokenization


def format_summary(summary):
    """Function formatting summary sheet."""
    formatted_summary = "Summary Report\n"
    formatted_summary += "-----------------------\n"
    for key, value in summary.items():
        if isinstance(value, float):
            # Format the float to two decimal places
            formatted_summary += f"{key}: {value:0.2f}\n"
        else:
            formatted_summary += f"{key}: {value}\n"
    return formatted_summary


class VectorStoreSummarizer:
    """Extract metrics from index. Summarize and write to text."""

    def __init__(self, index):
        self.index = index
        self.ref_doc_info = index.ref_doc_info
        self.docstore_info = index.docstore.docs
        self.node_df = (
            self.create_node_df()
        )  # Preprocess and store for use in other methods.

    def create_node_df(self):
        """Preprocess data once separate leaf & root nodes, and to be used in other methods."""
        node_ids_metadata = [
            {
                "node_id": key,
                "node_type": (
                    "leaf node"
                    if (
                        value.metadata
                        and isinstance(value.metadata, dict)
                        and value.metadata != {}
                    )
                    else "root node"
                ),
            }
            for key, value in self.docstore_info.items()
        ]
        return pd.DataFrame(node_ids_metadata)

    def get_node_ids(self, node_type="all", return_count=False):
        """Generalized method to get node IDs based on type."""
        if node_type in ["leaf node", "root node"]:
            filtered_ids = self.node_df[self.node_df["node_type"] == node_type][
                "node_id"
            ].tolist()
        else:
            filtered_ids = self.node_df["node_id"].tolist()

        return len(filtered_ids) if return_count else filtered_ids

    def get_texts(self, node_type_filter="all"):
        """Simplified get_texts method to merge text data with node types (leaf or root)."""
        node_id_text = [
            {"node_id": key, "text": value.text}
            for key, value in self.docstore_info.items()
        ]
        df_txt = pd.DataFrame(node_id_text)
        df_merged = pd.merge(self.node_df, df_txt, on="node_id", how="left")

        if node_type_filter in ["leaf node", "root node"]:
            df_filtered = df_merged[df_merged["node_type"] == node_type_filter]
        else:
            df_filtered = df_merged

        return df_filtered["text"].tolist()

    def create_summary(self):
        """
        Create a summary with counts of different node types and documents.

        :return: A dictionary containing summary metrics.
        """
        texts = self.get_texts(node_type_filter="leaf node")
        num_of_toks = 0
        for text in texts:
            doc = nlp(text)
            num_of_toks += len(doc)
        leaf_n_txt_size = num_of_toks

        texts = self.get_texts(node_type_filter="root node")
        num_of_toks = 0
        for text in texts:
            doc = nlp(text)
            num_of_toks += len(doc)
        root_n_txt_size = num_of_toks

        summary = {}
        num_of_leaf_n = len(self.get_node_ids(node_type="leaf node"))
        num_of_root_n = len(self.get_node_ids(node_type="root node"))

        # summary sheet
        summary["All Chunks of Text (nodes):"] = len(self.get_node_ids())
        summary["Chunks of Original Text (Leaf Nodes)"] = num_of_leaf_n
        summary["Sub-documents - LLM Generated Summaries (Root Nodes)"] = num_of_root_n

        # Adding error checking for division by zero
        summary["Average Chunks per Sub-document"] = (
            None if num_of_root_n == 0 else num_of_leaf_n / num_of_root_n
        )
        summary["Average Tokens per Chunk"] = (
            None if num_of_leaf_n == 0 else leaf_n_txt_size / num_of_leaf_n
        )
        summary["Average Tokens per LLM Generated Summary"] = (
            None if num_of_root_n == 0 else root_n_txt_size / num_of_root_n
        )

        return summary


# Count number of original document files ingested
DATA_DIR_PATH = r"C:\github\chatgpt\rag deloitte transparency reports\data\raw data"

f_paths = os.listdir(DATA_DIR_PATH)
no_of_doc_files = len(f_paths)


# Load index
# persist_dir = Path(
#     r"C:\github\chatgpt\rag deloitte transparency reports\index_summ_idx_md"
# )  # enter the directory/folder where the index files were persisted to

persist_dir = Path(
    r"C:\github\chatgpt\rag deloitte transparency reports\index_sent_window_md\sentence_index"
)

# Load index from disk
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
doc_summary_index = load_index_from_storage(storage_context)
print("Loading index done!")


# Instantiate ref_doc_info & docstore.docs
ref_doc_info = doc_summary_index.ref_doc_info
docstore_info = doc_summary_index.docstore.docs


# Get the summary
summarizer = VectorStoreSummarizer(doc_summary_index)
data_summary = summarizer.create_summary()


# format & print
formatted_summary = format_summary(data_summary)

# Add number of original docs ingested
formatted_summary += "-----------------------\n"
formatted_summary += f"Original Documents Loaded: {no_of_doc_files}"

print(formatted_summary)

# writing to a file in a pretty format
f_path = r"C:\github\chatgpt\rag deloitte transparency reports\data\index data summary\sent_window_summary_report.txt"
with open(f_path, "w") as f:
    f.write(formatted_summary)
