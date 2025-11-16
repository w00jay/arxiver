import json
import logging
import sqlite3

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_batch(paper_ids, vdb_path=None):
    """
    Get embeddings for multiple paper IDs in a single batch query.

    Args:
        paper_ids: List of paper IDs to get embeddings for
        vdb_path: Deprecated - uses chromadb_manager instead

    Returns:
        dict mapping paper_id -> numpy array of embedding (or None if not found)
    """
    result = {}
    if not paper_ids:
        return result

    try:
        from .chromadb_manager import chromadb_manager

        with chromadb_manager.get_collection_context(allow_concurrent=False) as vectors:
            # Batch query all paper IDs at once
            res = vectors.get(ids=paper_ids, include=["embeddings"])

            if res and res.get("ids") is not None and res.get("embeddings") is not None:
                # Map IDs to embeddings
                for idx, paper_id in enumerate(res["ids"]):
                    embedding = res["embeddings"][idx]
                    if embedding is not None:
                        embedding_array = np.array(embedding)
                        if embedding_array.size > 0:
                            result[paper_id] = embedding_array
                        else:
                            result[paper_id] = None
                    else:
                        result[paper_id] = None

            # For any IDs not found, set to None
            for paper_id in paper_ids:
                if paper_id not in result:
                    result[paper_id] = None

    except Exception as e:
        logger.error(f"Error accessing ChromaDB in batch: {e}")
        # Return None for all IDs on error
        for paper_id in paper_ids:
            result[paper_id] = None

    return result


def get_embedding(paper_id, vdb_path=None):
    """
    Get embedding for a paper ID using ChromaDB manager for consistency.

    Args:
        paper_id: Paper ID to get embedding for
        vdb_path: Deprecated - uses chromadb_manager instead

    Returns:
        numpy array of embedding or None if not found/error
    """
    try:
        # Use the ChromaDB manager for consistent access
        from .chromadb_manager import chromadb_manager

        # Get collection using the manager (without forcing new collection each time)
        with chromadb_manager.get_collection_context(allow_concurrent=False) as vectors:
            # Try to get the embedding
            res = vectors.get(ids=[paper_id], include=["embeddings"])
            if res and res.get("embeddings") is not None and len(res["embeddings"]) > 0:
                embedding = res["embeddings"][0]
                # Fix: Safely check if embedding exists and has data
                if embedding is not None:
                    # Convert to numpy array and check if it has elements
                    embedding_array = np.array(embedding)
                    if embedding_array.size > 0:
                        return embedding_array

            return None

    except Exception as e:
        logger.error(f"Error accessing ChromaDB for {paper_id}: {e}")
        return None
