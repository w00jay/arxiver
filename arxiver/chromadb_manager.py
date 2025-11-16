#!/usr/bin/env python3
"""
ChromaDB connection manager with proper resource management and error handling.
"""

import atexit
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Singleton ChromaDB manager with proper resource management."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "./data/arxiv_embeddings.chroma"):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.db_path = db_path
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._initialized = True

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def get_client(self):
        """Get or create ChromaDB client with error handling."""
        if self._client is None:
            try:
                logger.info(f"Creating ChromaDB client at: {self.db_path}")
                self._client = chromadb.PersistentClient(path=self.db_path)
                logger.info("ChromaDB client created successfully")
            except Exception as e:
                logger.error(f"Failed to create ChromaDB client: {e}")
                raise
        return self._client

    def get_embedding_function(self):
        """Get or create embedding function."""
        if self._embedding_function is None:
            try:
                logger.info("Creating sentence transformer embedding function...")
                self._embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
                logger.info("Embedding function created successfully")
            except Exception as e:
                logger.error(f"Failed to create embedding function: {e}")
                raise
        return self._embedding_function

    def get_collection(self, collection_name: str = "arxiver", force_new: bool = False):
        """Get or create collection with proper error handling."""
        if self._collection is None or force_new:
            try:
                client = self.get_client()
                embedding_function = self.get_embedding_function()

                # Try to get existing collection first
                try:
                    logger.info(
                        f"Attempting to get existing collection: {collection_name}"
                    )
                    self._collection = client.get_collection(name=collection_name)
                    logger.info(
                        f"Got existing collection with {self._collection.count()} documents"
                    )
                except Exception:
                    logger.info(
                        f"Collection not found, creating new one: {collection_name}"
                    )
                    self._collection = client.create_collection(
                        name=collection_name,
                        embedding_function=embedding_function,
                        metadata={"hnsw:space": "cosine"},
                    )
                    logger.info("Created new collection successfully")

            except Exception as e:
                logger.error(f"Failed to get/create collection: {e}")
                raise

        return self._collection

    def cleanup(self):
        """Clean up resources."""
        if self._client:
            try:
                # ChromaDB doesn't have explicit close method, but we can clear references
                self._client = None
                self._collection = None
                self._embedding_function = None
                logger.info("ChromaDB resources cleaned up")
            except Exception as e:
                logger.error(f"Error during ChromaDB cleanup: {e}")

    @contextmanager
    def get_collection_context(
        self, collection_name: str = "arxiver", allow_concurrent: bool = False
    ):
        """Context manager for safe collection access."""
        try:
            # For concurrent operations, get a fresh collection instance
            collection = self.get_collection(
                collection_name, force_new=allow_concurrent
            )
            yield collection
        except Exception as e:
            logger.error(f"Error in collection context: {e}")
            raise
        # Note: ChromaDB doesn't need explicit closing, but we could add retry logic here

    def health_check(self) -> bool:
        """Check if ChromaDB is healthy and accessible."""
        try:
            logger.info(f"Health check using db_path: {self.db_path}")
            client = self.get_client()
            logger.info(f"Client obtained: {client}")
            collections = client.list_collections()
            logger.info(
                f"Health check passed. Collections: {[c.name for c in collections]}"
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            import traceback

            logger.error(f"Health check traceback:\n{traceback.format_exc()}")
            return False

    def reset_connection(self):
        """Reset connection in case of corruption."""
        logger.warning("Resetting ChromaDB connection")
        self.cleanup()
        self._initialized = False
        self.__init__(self.db_path)


# Compute correct path for global instance
# This ensures the path is correct regardless of where the script is run from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_EMBEDDINGS_PATH = os.path.join(
    _PROJECT_ROOT, "data", "arxiv_embeddings.chroma"
)

# Global instance with correct absolute path
chromadb_manager = ChromaDBManager(db_path=_DEFAULT_EMBEDDINGS_PATH)
