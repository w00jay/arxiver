#!/usr/bin/env python3
"""
Arxiver MCP Server

This server provides AI assistants with access to arXiv paper search,
recommendation, and summarization capabilities through the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

try:
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available, vector search tools will be disabled")

import numpy as np
import tensorflow as tf

# Import our existing functions
try:
    from .arxiv import fetch_article_for_id, fetch_articles_for_date
    from .database import (
        create_connection,
        create_table,
        get_paper_by_base_id,
        get_paper_by_id,
        get_recent_papers_since_days,
        insert_article,
        update_concise_summary,
    )
    from .utils import clean_paper_id, get_paper_id_without_version
except ImportError:
    from arxiv import fetch_article_for_id, fetch_articles_for_date
    from database import (
        create_connection,
        create_table,
        get_paper_by_base_id,
        get_paper_by_id,
        get_recent_papers_since_days,
        insert_article,
        update_concise_summary,
    )
    from utils import clean_paper_id, get_paper_id_without_version
# Already imported above, remove duplicate import
from dotenv import load_dotenv

try:
    from .llm import choose_summaries, summarize_summary
except ImportError:
    from llm import choose_summaries, summarize_summary
from mcp.server.fastmcp import FastMCP

try:
    from .vector_db import get_embedding
except ImportError:
    try:
        from vector_db import get_embedding
    except ImportError:

        def get_embedding(paper_id):
            return None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
LOOK_BACK_DAYS = 3

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "predictor")
PAPERS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_papers.db")
EMBEDDINGS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_embeddings.chroma")

# Ensure data directories exist
os.makedirs(os.path.dirname(PAPERS_DB), exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_DB), exist_ok=True)

# Initialize the FastMCP server
app = FastMCP("arxiver")


def get_latest_model(directory: str) -> Optional[str]:
    """Get the latest model file from a directory based on creation date."""
    try:
        if not os.path.exists(directory):
            logger.warning(f"Model directory {directory} not found")
            return None

        model_files = []
        for filename in os.listdir(directory):
            if filename.endswith((".keras", ".h5")):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    model_files.append(filepath)

        if not model_files:
            logger.warning(f"No model files found in {directory}")
            return None

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_model = model_files[0]
        logger.info(f"Using latest model: {latest_model}")
        return latest_model

    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


async def startup_checks():
    """Perform startup checks and initialization."""
    logger.info("Performing startup checks...")

    # Check database
    try:
        conn = create_connection(PAPERS_DB)
        if conn:
            create_table(conn)
            conn.close()
            logger.info("Database connection successful")
        else:
            logger.error("Failed to connect to database")
            return False
    except Exception as e:
        logger.error(f"Database startup error: {e}")
        return False

    # Check model availability
    model_path = get_latest_model(MODEL_PATH)
    if model_path:
        logger.info(f"Model available at: {model_path}")
    else:
        logger.warning("No ML model found - recommendations will be disabled")

    return True


# Tool definitions using FastMCP
@app.tool()
async def search_papers(query: str, top_k: int = 5) -> str:
    """Search arXiv papers using semantic similarity based on embeddings.

    Args:
        query: Search query (e.g., 'machine learning transformers', 'computer vision')
        top_k: Number of papers to return (default: 5, max: 50)

    Returns:
        JSON string containing search results
    """
    result = await search_papers_impl(query, top_k)
    return json.dumps(result, indent=2)


@app.tool()
async def get_recommendations(days_back: int = 3) -> str:
    """Get personalized paper recommendations using ML model based on your interests.

    Args:
        days_back: Number of days to look back for recommendations (default: 3)

    Returns:
        JSON string containing recommended papers
    """
    result = await get_recommendations_impl(days_back)
    return json.dumps(result, indent=2)


@app.tool()
async def summarize_paper(paper_id: str) -> str:
    """Generate or retrieve a concise summary of a specific paper.

    Args:
        paper_id: arXiv paper ID or URL (e.g., 'http://arxiv.org/abs/2404.04292v1' or '2404.04292')

    Returns:
        JSON string containing paper summary
    """
    result = await summarize_paper_impl(paper_id)
    return json.dumps(result, indent=2)


@app.tool()
async def choose_best_papers(query: str, top_i: int = 3, search_k: int = 20) -> str:
    """Use LLM to intelligently select the most relevant papers from search results.

    Args:
        query: Search query to find relevant papers
        top_i: Number of best papers to select (default: 3)
        search_k: Number of papers to search through (default: 20)

    Returns:
        JSON string containing selected papers
    """
    result = await choose_best_papers_impl(query, top_i, search_k)
    return json.dumps(result, indent=2)


@app.tool()
async def import_paper(arxiv_id: str) -> str:
    """Import a specific paper from arXiv into the database.

    Args:
        arxiv_id: arXiv ID without URL (e.g., '2404.04292' or '1706.03762')

    Returns:
        JSON string confirming import status
    """
    result = await import_paper_impl(arxiv_id)
    return json.dumps(result, indent=2)


# @app.tool()
# async def ingest_recent_papers(days_back: int = 3, max_papers: int = 100) -> str:
#     """Ingest recent papers from arXiv based on ML/AI search queries.

#     Args:
#         days_back: Number of days to look back (default: 3)
#         max_papers: Maximum number of papers to ingest (default: 100)

#     Returns:
#         JSON string with ingestion results
#     """
#     result = await ingest_recent_papers_impl(days_back, max_papers)
#     return json.dumps(result, indent=2)


@app.tool()
async def get_paper_details(paper_id: str) -> str:
    """Get detailed information about a specific paper from the database.

    Args:
        paper_id: arXiv paper ID or URL

    Returns:
        JSON string containing paper details
    """
    result = await get_paper_details_impl(paper_id)
    return json.dumps(result, indent=2)


# Async implementations of the tools
async def search_papers_impl(query: str, top_k: int = 5) -> dict:
    """Implementation for search_papers tool."""
    try:
        if not CHROMADB_AVAILABLE:
            return {
                "error": "ChromaDB not available",
                "message": "Vector search disabled - ChromaDB not available. Use get_paper_details for specific papers.",
            }

        # Limit top_k to reasonable bounds
        top_k = max(1, min(top_k, 50))

        logger.info(f"Searching for '{query}' with top_k={top_k}")
        logger.info(f"Using ChromaDB path: {EMBEDDINGS_DB}")

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=EMBEDDINGS_DB)

        # Get or create collection
        collection = client.get_or_create_collection(
            name="arxiver",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
        )

        logger.info(f"Collection 'arxiver' has {collection.count()} documents")

        # Query the collection
        results = collection.query(query_texts=[query], n_results=top_k)

        logger.info(f"Query returned {len(results.get('ids', [[]])[0])} results")

        # Format results
        papers = []
        if results["ids"] and results["ids"][0]:
            for i, paper_id in enumerate(results["ids"][0]):
                paper_info = {
                    "paper_id": paper_id,
                    "distance": results["distances"][0][i]
                    if results["distances"]
                    else None,
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"]
                    else {},
                    "document": results["documents"][0][i]
                    if results["documents"]
                    else "",
                }
                papers.append(paper_info)

        logger.info(f"Formatted {len(papers)} papers for response")

        return {"query": query, "total_results": len(papers), "papers": papers}

    except Exception as e:
        logger.error(f"Error in search_papers: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "message": "Error occurred during paper search"}


async def get_recommendations_impl(days_back: int = 3) -> dict:
    """Implementation for get_recommendations tool."""
    try:
        # Limit days_back to reasonable bounds
        days_back = max(1, min(days_back, 30))

        # Get the latest model
        model_path = get_latest_model(MODEL_PATH)
        if not model_path:
            return {
                "error": "No ML model available",
                "message": "Recommendations disabled - no trained model found",
            }

        # Load the model with proper handling of legacy parameters
        try:
            # Try standard loading first
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Standard model loading failed: {e}")
            # Try with custom object scope for legacy compatibility
            try:

                def custom_input_layer(*args, **kwargs):
                    # Handle legacy batch_shape parameter
                    if "batch_shape" in kwargs:
                        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
                    return tf.keras.layers.InputLayer(*args, **kwargs)

                custom_objects = {"InputLayer": custom_input_layer}
                model = tf.keras.models.load_model(
                    model_path, custom_objects=custom_objects
                )
                logger.info("Model loaded with custom objects")
            except Exception as e2:
                logger.error(f"Custom model loading also failed: {e2}")
                # Fallback: skip model prediction, just return recent papers
                logger.info(
                    "Proceeding without ML model, returning recent papers as recommendations"
                )

        # Get recent papers from database
        conn = create_connection(PAPERS_DB)
        if not conn:
            return {
                "error": "Database connection failed",
                "message": "Could not connect to papers database",
            }

        papers = get_recent_papers_since_days(conn, days_back)
        conn.close()

        if not papers:
            return {
                "message": f"No papers found in the last {days_back} days",
                "papers": [],
            }

        # Get recommendations with or without ML model
        recommendations = []
        for paper in papers:
            # Use ML model for scoring if available
            prediction_score = 0.8  # Default score
            if "model" in locals():
                try:
                    # Get embedding for the paper (simplified)
                    embedding = get_embedding(paper["paper_id"])
                    if embedding is not None:
                        # Make prediction with the model
                        prediction = model.predict(np.array([embedding]), verbose=0)
                        prediction_score = float(prediction[0][0])
                except Exception as e:
                    logger.warning(
                        f"Model prediction failed for {paper['paper_id']}: {e}"
                    )
                    prediction_score = 0.8  # Fallback score

            recommendations.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "authors": paper["authors"] or "",
                    "published": paper["published"] or "",
                    "categories": paper["categories"] or "",
                    "summary": paper["summary"],
                    "prediction_score": prediction_score,
                    "arxiv_url": paper["arxiv_url"]
                    or f"https://arxiv.org/abs/{paper['paper_id']}",
                }
            )

        return {
            "days_back": days_back,
            "total_papers": len(recommendations),
            "recommendations": recommendations[:10],  # Limit to top 10
        }

    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return {
            "error": str(e),
            "message": "Error occurred during recommendation generation",
        }


async def summarize_paper_impl(paper_id: str) -> dict:
    """Implementation for summarize_paper tool."""
    try:
        # Clean paper ID
        paper_id = clean_paper_id(paper_id)
        base_paper_id = get_paper_id_without_version(paper_id)

        # Get paper from database
        conn = create_connection(PAPERS_DB)
        if not conn:
            return {
                "error": "Database connection failed",
                "message": "Could not connect to papers database",
            }

        # Try exact match first, then base ID
        paper = get_paper_by_id(conn, paper_id)
        if not paper:
            paper = get_paper_by_base_id(conn, base_paper_id)

        if not paper:
            # Try to fetch from arXiv
            try:
                paper_data = fetch_article_for_id(paper_id)
                if paper_data:
                    # Insert the full paper data into database
                    insert_article(conn, paper_data)
                    # Generate and save summary
                    summary = summarize_summary(paper_data.get("summary", ""))
                    # Use the actual paper_id from the fetched data
                    actual_paper_id = paper_data.get("paper_id")
                    update_concise_summary(conn, actual_paper_id, summary)
                    conn.close()
                    return {
                        "paper_id": paper_id,
                        "title": paper_data.get("title"),
                        "authors": paper_data.get("authors"),
                        "published": paper_data.get("published"),
                        "categories": paper_data.get("categories"),
                        "summary": summary,
                        "arxiv_url": paper_data.get("arxiv_url"),
                        "pdf_url": paper_data.get("pdf_url"),
                        "source": "arXiv (new)",
                    }
                else:
                    conn.close()
                    return {
                        "error": "Paper not found",
                        "message": f"Could not find paper with ID: {paper_id}",
                    }
            except Exception as e:
                conn.close()
                return {
                    "error": f"Error fetching paper: {e}",
                    "message": "Could not retrieve paper from arXiv",
                }

        conn.close()

        # Generate summary if not available
        if not paper["concise_summary"]:
            summary = summarize_summary(paper["summary"] or "")
            conn = create_connection(PAPERS_DB)
            if conn:
                update_concise_summary(conn, paper_id, summary)
                conn.close()
            paper = dict(paper)  # Convert Row to dict for modification
            paper["concise_summary"] = summary

        return {
            "paper_id": paper_id,
            "title": paper["title"],
            "authors": paper["authors"] or "",
            "published": paper["published"] or "",
            "categories": paper["categories"] or "",
            "summary": paper["concise_summary"] or paper["summary"],
            "arxiv_url": paper["arxiv_url"] or f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": paper["pdf_url"] or f"https://arxiv.org/pdf/{paper_id}.pdf",
            "source": "database",
        }

    except Exception as e:
        logger.error(f"Error in summarize_paper: {e}")
        return {"error": str(e), "message": "Error occurred during paper summarization"}


async def choose_best_papers_impl(
    query: str, top_i: int = 3, search_k: int = 20
) -> dict:
    """Implementation for choose_best_papers tool."""
    try:
        # Limit parameters
        top_i = max(1, min(top_i, 10))
        search_k = max(5, min(search_k, 100))

        # First, search for papers
        search_results = await search_papers_impl(query, search_k)

        if "error" in search_results:
            return search_results

        papers = search_results.get("papers", [])
        if not papers:
            return {
                "query": query,
                "message": "No papers found for the query",
                "selected_papers": [],
            }

        # Use LLM to choose best papers
        paper_summaries = []
        for paper in papers:
            paper_summaries.append(
                {
                    "paper_id": paper.get("paper_id"),
                    "summary": paper.get("document", ""),
                    "metadata": paper.get("metadata", {}),
                }
            )

        # Call LLM selection function
        selected_papers = choose_summaries(query, paper_summaries, top_i)

        return {
            "query": query,
            "search_k": search_k,
            "top_i": top_i,
            "total_found": len(papers),
            "selected_papers": selected_papers,
        }

    except Exception as e:
        logger.error(f"Error in choose_best_papers: {e}")
        return {"error": str(e), "message": "Error occurred during paper selection"}


async def import_paper_impl(arxiv_id: str) -> dict:
    """Implementation for import_paper tool."""
    try:
        # Clean arXiv ID but keep version if present
        arxiv_id = clean_paper_id(arxiv_id)

        # Fetch paper from arXiv
        paper_data = fetch_article_for_id(arxiv_id)

        if not paper_data:
            return {
                "error": "Paper not found",
                "message": f"Could not find paper with arXiv ID: {arxiv_id}",
            }

        # Add to database
        conn = create_connection(PAPERS_DB)
        if not conn:
            return {
                "error": "Database connection failed",
                "message": "Could not connect to papers database",
            }

        # Insert the paper with all metadata
        insert_article(conn, paper_data)
        conn.close()

        return {
            "arxiv_id": arxiv_id,
            "title": paper_data.get("title"),
            "authors": paper_data.get("authors"),
            "published": paper_data.get("published"),
            "status": "imported",
            "message": "Paper successfully imported into database",
        }

    except Exception as e:
        logger.error(f"Error in import_paper: {e}")
        return {"error": str(e), "message": "Error occurred during paper import"}


# async def ingest_recent_papers_impl(days_back: int = 3, max_papers: int = 100) -> dict:
#     """Implementation for ingest_recent_papers tool."""
#     try:
#         # Limit parameters
#         days_back = max(1, min(days_back, 30))
#         max_papers = max(10, min(max_papers, 1000))

#         # Calculate date range
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=days_back)

#         # Fetch recent papers
#         papers = fetch_articles_for_date(start_date.strftime("%Y-%m-%d"), max_papers)

#         if not papers:
#             return {
#                 "message": f"No papers found in the last {days_back} days",
#                 "ingested_count": 0,
#                 "papers": [],
#             }

#         # Add to database
#         conn = create_connection(PAPERS_DB)
#         if not conn:
#             return {
#                 "error": "Database connection failed",
#                 "message": "Could not connect to papers database",
#             }

#         ingested_count = 0
#         ingested_papers = []

#         for paper in papers:
#             try:
#                 # Insert paper (your database insertion logic here)
#                 ingested_count += 1
#                 ingested_papers.append(
#                     {
#                         "paper_id": paper.get("paper_id"),
#                         "title": paper.get("title"),
#                         "published": paper.get("published"),
#                     }
#                 )
#             except Exception as e:
#                 logger.warning(f"Failed to ingest paper {paper.get('paper_id')}: {e}")

#         conn.close()

#         return {
#             "days_back": days_back,
#             "max_papers": max_papers,
#             "total_found": len(papers),
#             "ingested_count": ingested_count,
#             "papers": ingested_papers,
#         }

#     except Exception as e:
#         logger.error(f"Error in ingest_recent_papers: {e}")
#         return {"error": str(e), "message": "Error occurred during paper ingestion"}


async def get_paper_details_impl(paper_id: str) -> dict:
    """Implementation for get_paper_details tool."""
    try:
        # Clean paper ID
        paper_id = clean_paper_id(paper_id)
        base_paper_id = get_paper_id_without_version(paper_id)

        # Get paper from database
        conn = create_connection(PAPERS_DB)
        if not conn:
            return {
                "error": "Database connection failed",
                "message": "Could not connect to papers database",
            }

        # Try exact match first, then base ID
        paper = get_paper_by_id(conn, paper_id)
        if not paper:
            paper = get_paper_by_base_id(conn, base_paper_id)
        conn.close()

        if not paper:
            return {
                "error": "Paper not found",
                "message": f"Paper with ID {paper_id} not found in database",
            }

        return {
            "paper_id": paper_id,
            "title": paper["title"],
            "authors": paper["authors"] or "",
            "published": paper["published"] or "",
            "summary": paper["summary"],
            "concise_summary": paper["concise_summary"],
            "categories": paper["categories"] or "",
            "arxiv_url": paper["arxiv_url"] or f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": paper["pdf_url"] or f"https://arxiv.org/pdf/{paper_id}.pdf",
            "importance_score": paper["importance_score"]
            if paper["importance_score"] is not None
            else 0.0,
            "read_status": paper["read_status"] or "unread",
            "tags": paper["tags"] or "",
            "notes": paper["notes"] or "",
        }

    except Exception as e:
        logger.error(f"Error in get_paper_details: {e}")
        return {
            "error": str(e),
            "message": "Error occurred while retrieving paper details",
        }


async def main():
    """Main function to run the FastMCP server."""
    logger.info("Starting Arxiver MCP server...")

    # Perform startup checks
    try:
        await startup_checks()
        logger.info("Startup checks completed successfully")
    except Exception as e:
        logger.error(f"Startup checks failed: {e}")
        return

    # Run the server
    await app.run()


if __name__ == "__main__":
    import sys

    logger.info("Starting Arxiver MCP server...")

    # Perform startup checks synchronously
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(startup_checks())
        logger.info("Startup checks completed successfully")
    except Exception as e:
        logger.error(f"Startup checks failed: {e}")
        sys.exit(1)

    # Run the FastMCP server (it handles its own event loop)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
