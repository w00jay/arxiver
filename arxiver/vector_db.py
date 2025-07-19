import json
import logging
import sqlite3

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding(paper_id, vdb_path="./data/arxiv_embeddings.chroma"):
    """
    Get embedding for a paper ID using ChromaDB manager for consistency.

    Args:
        paper_id: Paper ID to get embedding for
        vdb_path: Path to ChromaDB data (for compatibility, but uses manager)

    Returns:
        numpy array of embedding or None if not found/error
    """
    try:
        # Use the ChromaDB manager for consistent access
        from chromadb_manager import chromadb_manager

        # Get collection using the manager (with concurrent access)
        with chromadb_manager.get_collection_context(allow_concurrent=True) as vectors:
            # Try to get the embedding
            res = vectors.get(ids=[paper_id], include=["embeddings"])
            if res and res.get("embeddings") and len(res["embeddings"]) > 0:
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
