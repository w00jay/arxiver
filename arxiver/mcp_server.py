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

# from pydantic import AnyUrl  # Not needed for basic functionality
# Import our existing functions
from arxiv import fetch_article_for_id, fetch_articles_for_date
from database import (
    create_connection,
    create_table,
    get_paper_by_id,
    get_recent_papers_since_days,
    update_concise_summary,
)
from dotenv import load_dotenv
from llm import choose_summaries, summarize_summary
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

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

# Initialize the MCP server
app = Server("arxiver")


def get_latest_model(directory: str) -> Optional[str]:
    """Get the latest model file from a directory based on creation date."""
    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".keras")
        ]
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None


def get_chromadb_collection():
    """Get or create ChromaDB collection for embeddings."""
    if not CHROMADB_AVAILABLE:
        return None
    try:
        vdb = chromadb.PersistentClient(path=EMBEDDINGS_DB)
        sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        vectors = vdb.get_or_create_collection(
            name="arxiver", embedding_function=sentence_transformer_ef
        )
        return vectors
    except Exception as e:
        logger.error(f"Error getting ChromaDB collection: {e}")
        return None


# Tool definitions
@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_papers",
            description="Search arXiv papers using semantic similarity based on embeddings"
            + ("" if CHROMADB_AVAILABLE else " (DISABLED - ChromaDB not available)"),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'machine learning transformers', 'computer vision')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of papers to return (default: 5, max: 50)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        )
        if CHROMADB_AVAILABLE
        else Tool(
            name="search_papers_disabled",
            description="Vector search disabled - ChromaDB not available. Use get_paper_details for specific papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Informational message",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_recommendations",
            description="Get personalized paper recommendations using ML model based on your interests",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to look back for recommendations (default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="summarize_paper",
            description="Generate or retrieve a concise summary of a specific paper",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID or URL (e.g., 'http://arxiv.org/abs/2404.04292v1' or '2404.04292')",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="choose_best_papers",
            description="Use LLM to intelligently select the most relevant papers from search results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant papers",
                    },
                    "top_i": {
                        "type": "integer",
                        "description": "Number of best papers to select (default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "search_k": {
                        "type": "integer",
                        "description": "Number of papers to search through (default: 20)",
                        "default": 20,
                        "minimum": 5,
                        "maximum": 100,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="import_paper",
            description="Import a specific paper from arXiv into the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv ID without URL (e.g., '2404.04292' or '1706.03762')",
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        Tool(
            name="ingest_recent_papers",
            description="Ingest recent papers from arXiv based on ML/AI search queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 14,
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (default: today)",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_paper_details",
            description="Get detailed information about a specific paper from the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID or URL",
                    },
                },
                "required": ["paper_id"],
            },
        ),
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "search_papers":
            result = await search_papers(arguments)
        elif name == "get_recommendations":
            result = await get_recommendations(arguments)
        elif name == "summarize_paper":
            result = await summarize_paper(arguments)
        elif name == "choose_best_papers":
            result = await choose_best_papers(arguments)
        elif name == "import_paper":
            result = await import_paper(arguments)
        elif name == "ingest_recent_papers":
            result = await ingest_recent_papers(arguments)
        elif name == "get_paper_details":
            result = await get_paper_details(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


# Tool implementations
async def search_papers(arguments: dict) -> dict:
    """Search papers using vector similarity."""
    if not CHROMADB_AVAILABLE:
        return {
            "error": "Vector search not available - ChromaDB not installed",
            "suggestion": "Use get_paper_details to get information about specific papers by ID",
        }

    query = arguments["query"]
    top_k = arguments.get("top_k", 5)

    logger.info(f"Searching papers for query: '{query}' (top_k: {top_k})")

    vectors = get_chromadb_collection()
    if not vectors:
        raise Exception("Could not connect to vector database")

    results = vectors.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    papers = []
    for i in range(len(results["ids"][0])):
        paper = {
            "paper_id": results["ids"][0][i],
            "summary": results["documents"][0][i],
            "similarity_score": 1
            - results["distances"][0][i],  # Convert distance to similarity
            "metadata": results["metadatas"][0][i],
        }
        papers.append(paper)

    return {
        "query": query,
        "papers_found": len(papers),
        "papers": papers,
    }


async def get_recommendations(arguments: dict) -> dict:
    """Get ML-based paper recommendations."""
    days_back = arguments.get("days_back", 3)

    logger.info(f"Getting recommendations for last {days_back} days")

    # Load the latest model
    latest_model_path = get_latest_model(MODEL_PATH)
    if not latest_model_path:
        raise Exception("No ML model found for recommendations")

    model = tf.keras.models.load_model(latest_model_path, compile=False)
    logger.info(f"Loaded model: {latest_model_path}")

    # Get recent papers
    conn = create_connection(PAPERS_DB)
    if not conn:
        raise Exception("Could not connect to papers database")

    recent_papers = get_recent_papers_since_days(conn, days=days_back)
    logger.info(f"Found {len(recent_papers)} recent papers")

    # Get embeddings and make predictions
    recommendations = []
    new_X = []
    valid_papers = []

    for paper in recent_papers:
        paper_id, title, summary, concise_summary = paper[:4]  # Handle extra fields
        embedding = get_embedding(paper_id)
        if embedding is not None:
            new_X.append(embedding)
            valid_papers.append(
                {
                    "paper_id": paper_id,
                    "title": title.replace("\n", ""),
                    "summary": summary,
                    "concise_summary": concise_summary,
                }
            )

    if not new_X:
        return {
            "days_back": days_back,
            "recommendations": [],
            "message": "No papers with embeddings found for the specified period",
        }

    # Make predictions
    new_X = np.array(new_X)
    predictions = model.predict(new_X) > 0.5

    for i, is_recommended in enumerate(predictions):
        if is_recommended:
            paper = valid_papers[i]
            recommendations.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "summary": paper["concise_summary"] or paper["summary"],
                    "arxiv_url": paper["paper_id"],
                    "pdf_url": paper["paper_id"].replace("abs", "pdf"),
                }
            )

    conn.close()

    return {
        "days_back": days_back,
        "total_papers_analyzed": len(valid_papers),
        "recommendations_count": len(recommendations),
        "recommendations": recommendations,
    }


async def summarize_paper(arguments: dict) -> dict:
    """Generate or retrieve paper summary."""
    paper_id = arguments["paper_id"]

    logger.info(f"Summarizing paper: {paper_id}")

    # Normalize paper ID
    if not paper_id.startswith("http"):
        paper_id = f"http://arxiv.org/abs/{paper_id}"

    conn = create_connection(PAPERS_DB)
    if not conn:
        raise Exception("Could not connect to papers database")

    try:
        paper = get_paper_by_id(conn, paper_id)
        if not paper:
            raise Exception(f"Paper not found: {paper_id}")

        paper_id_db, title, summary, concise_summary = paper

        if concise_summary:
            result = {
                "paper_id": paper_id_db,
                "title": title,
                "original_summary": summary,
                "concise_summary": concise_summary,
                "status": "existing_summary",
            }
        else:
            # Generate new summary
            logger.info("Generating new concise summary")
            concise_summary = summarize_summary(summary)
            update_concise_summary(conn, paper_id_db, concise_summary)

            result = {
                "paper_id": paper_id_db,
                "title": title,
                "original_summary": summary,
                "concise_summary": concise_summary,
                "status": "new_summary_generated",
            }

        return result

    finally:
        conn.close()


async def choose_best_papers(arguments: dict) -> dict:
    """Use LLM to choose the best papers from search results."""
    query = arguments["query"]
    top_i = arguments.get("top_i", 3)
    search_k = arguments.get("search_k", 20)

    logger.info(
        f"Choosing {top_i} best papers from {search_k} results for query: '{query}'"
    )

    # First, search for papers
    search_results = await search_papers({"query": query, "top_k": search_k})

    if not search_results["papers"]:
        return {
            "query": query,
            "selected_papers": [],
            "message": "No papers found for the given query",
        }

    # Use LLM to choose the best papers
    papers_data = search_results["papers"]
    chosen_papers = choose_summaries(papers_data, top_i)

    return {
        "query": query,
        "papers_searched": len(papers_data),
        "papers_selected": len(chosen_papers) if isinstance(chosen_papers, list) else 0,
        "selected_papers": chosen_papers,
    }


async def import_paper(arguments: dict) -> dict:
    """Import a specific paper from arXiv."""
    arxiv_id = arguments["arxiv_id"]

    logger.info(f"Importing paper: {arxiv_id}")

    conn = create_connection(PAPERS_DB)
    if not conn:
        raise Exception("Could not connect to papers database")

    try:
        create_table(conn)
        paper_id = fetch_article_for_id(conn, arxiv_id)

        if paper_id:
            # Generate summary
            paper = get_paper_by_id(conn, paper_id)
            if paper:
                paper_id_db, title, summary, concise_summary = paper

                if not concise_summary:
                    concise_summary = summarize_summary(summary)
                    update_concise_summary(conn, paper_id_db, concise_summary)

                return {
                    "arxiv_id": arxiv_id,
                    "paper_id": paper_id_db,
                    "title": title,
                    "summary": concise_summary,
                    "status": "imported_successfully",
                }

        raise Exception(f"Failed to import paper: {arxiv_id}")

    finally:
        conn.close()


async def ingest_recent_papers(arguments: dict) -> dict:
    """Ingest recent papers from arXiv."""
    days = arguments.get("days", 3)
    start_date = arguments.get("start_date")

    logger.info(f"Ingesting papers for {days} days from {start_date or 'today'}")

    conn = create_connection(PAPERS_DB)
    if not conn:
        raise Exception("Could not connect to papers database")

    try:
        create_table(conn)

        # Define search query for ML/AI papers
        search_query = (
            r'(all:"machine learning" OR all:"deep learning" OR all:"supervised learning" '
            r'OR all:"unsupervised learning" OR all:"reinforcement learning") OR '
            r'(all:"artificial intelligence" OR all:"natural language processing" OR all:"computer vision" '
            r'OR all:"robotics" OR all:"knowledge representation" OR all:"search algorithms") OR '
            r'(all:"large language models" OR all:"transformers" OR all:"GPT" OR all:"BERT" '
            r'OR all:"few-shot learning" OR all:"zero-shot learning") OR '
            r'(all:"data science" OR all:"ethics in AI" OR all:"AI in healthcare" OR all:"AI and society")'
        )

        # Parse start date
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_date_obj = datetime.now()

        # Fetch articles
        total_articles = 0
        for day_offset in range(days):
            query_date = start_date_obj - timedelta(days=day_offset)
            article_ids = fetch_articles_for_date(conn, search_query, query_date, 1500)
            total_articles += len(article_ids)

            # Generate summaries for new articles
            for article_id in article_ids:
                try:
                    paper = get_paper_by_id(conn, article_id)
                    if paper and not paper[3]:  # If no concise summary exists
                        concise_summary = summarize_summary(paper[2])
                        update_concise_summary(conn, article_id, concise_summary)
                except Exception as e:
                    logger.error(f"Error processing article {article_id}: {e}")

        return {
            "days": days,
            "start_date": start_date_obj.strftime("%Y-%m-%d"),
            "total_articles_processed": total_articles,
            "status": "ingestion_completed",
        }

    finally:
        conn.close()


async def get_paper_details(arguments: dict) -> dict:
    """Get detailed information about a paper."""
    paper_id = arguments["paper_id"]

    logger.info(f"Getting paper details for: {paper_id}")

    # Normalize paper ID
    if not paper_id.startswith("http"):
        paper_id = f"http://arxiv.org/abs/{paper_id}"

    conn = create_connection(PAPERS_DB)
    if not conn:
        raise Exception("Could not connect to papers database")

    try:
        paper = get_paper_by_id(conn, paper_id)
        if not paper:
            raise Exception(f"Paper not found: {paper_id}")

        paper_id_db, title, summary, concise_summary = paper

        return {
            "paper_id": paper_id_db,
            "title": title,
            "abstract": summary,
            "concise_summary": concise_summary,
            "arxiv_url": paper_id_db,
            "pdf_url": paper_id_db.replace("abs", "pdf"),
            "status": "found",
        }

    finally:
        conn.close()


async def main():
    """Main function to run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="arxiver",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
