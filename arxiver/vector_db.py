import json
import sqlite3

import chromadb
import numpy as np
from chromadb.utils import embedding_functions


def get_embedding(paper_id, vdb_path="../data/arxiv_embeddings.chroma"):
    vdb = chromadb.PersistentClient(vdb_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    embedding_func = sentence_transformer_ef
    vectors = vdb.get_or_create_collection(
        name="arxiver", embedding_function=embedding_func
    )

    res = vectors.get(ids=[paper_id], limit=1, include=["embeddings"])
    # print(f'{res["ids"][0]} {res["embeddings"][0]}')
    return res["embeddings"][0]
