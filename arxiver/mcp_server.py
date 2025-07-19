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
from datetime import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator

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

# Simple in-memory cache with TTL
_cache = {}
_cache_timestamps = {}
CACHE_TTL = 300  # 5 minutes

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "predictor")
PAPERS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_papers.db")
EMBEDDINGS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_embeddings.chroma")

# Ensure data directories exist
os.makedirs(os.path.dirname(PAPERS_DB), exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_DB), exist_ok=True)


# Pydantic Models for Type Safety and Response Schemas
class Paper(BaseModel):
    """Structured representation of an arXiv paper."""

    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    authors: Optional[str] = Field(None, description="Paper authors")
    published: Optional[str] = Field(None, description="Publication date")
    categories: Optional[str] = Field(None, description="arXiv categories")
    summary: Optional[str] = Field(None, description="Paper abstract/summary")
    concise_summary: Optional[str] = Field(
        None, description="AI-generated concise summary"
    )
    arxiv_url: Optional[str] = Field(None, description="URL to arXiv page")

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Paper ID cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Response schema for paper search operations."""

    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    papers: List[Paper] = Field(
        default_factory=list, description="List of matching papers"
    )
    search_method: Optional[str] = Field(
        None, description="Method used for search (vector, text, etc.)"
    )
    execution_time_ms: Optional[float] = Field(
        None, ge=0, description="Execution time in milliseconds"
    )


class RecommendationResponse(BaseModel):
    """Response schema for paper recommendations."""

    recommendations: List[Paper] = Field(
        default_factory=list, description="Recommended papers"
    )
    total_recommendations: int = Field(
        ..., ge=0, description="Total number of recommendations"
    )
    criteria: Optional[str] = Field(None, description="Recommendation criteria used")
    model_used: Optional[str] = Field(
        None, description="ML model used for recommendations"
    )
    execution_time_ms: Optional[float] = Field(
        None, ge=0, description="Execution time in milliseconds"
    )


class SummaryResponse(BaseModel):
    """Response schema for paper summarization."""

    paper_id: str = Field(..., description="Paper identifier")
    summary: str = Field(..., description="Generated or existing summary")
    summary_type: str = Field(
        ..., description="Type of summary (existing, generated, concise)"
    )
    generated_at: Optional[str] = Field(
        None, description="Timestamp when summary was generated"
    )
    execution_time_ms: Optional[float] = Field(
        None, ge=0, description="Execution time in milliseconds"
    )


class PaperDetailsResponse(BaseModel):
    """Response schema for detailed paper information."""

    paper: Optional[Paper] = Field(None, description="Paper details if found")
    found: bool = Field(..., description="Whether the paper was found")
    execution_time_ms: Optional[float] = Field(
        None, ge=0, description="Execution time in milliseconds"
    )


class ImportResponse(BaseModel):
    """Response schema for paper import operations."""

    arxiv_id: str = Field(..., description="arXiv ID that was imported")
    success: bool = Field(..., description="Whether import was successful")
    paper: Optional[Paper] = Field(None, description="Imported paper details")
    message: str = Field(..., description="Status message")
    execution_time_ms: Optional[float] = Field(
        None, ge=0, description="Execution time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: str = Field(
        default_factory=lambda: dt.now().isoformat(), description="Error timestamp"
    )


# Middleware Framework for FastMCP Server
class MCPMiddleware:
    """Base class for MCP middleware."""
    
    async def before_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Called before each tool execution. Return error string to abort."""
        pass
    
    async def after_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Called after each tool execution."""
        pass
    
    async def before_resource_read(self, resource_uri: str) -> Optional[str]:
        """Called before each resource read. Return error string to abort."""
        pass
    
    async def after_resource_read(self, resource_uri: str, result: Any) -> None:
        """Called after each resource read."""
        pass


class LoggingMiddleware(MCPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self):
        self.request_count = 0
        self.start_time = datetime.now()
    
    async def before_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        self.request_count += 1
        # Log request with sanitized args (remove sensitive data)
        safe_args = self._sanitize_args(args)
        logger.info(f"Tool call #{self.request_count}: {tool_name}({safe_args})")
        return None
    
    async def after_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        # Log result size and type
        result_info = self._get_result_info(result)
        logger.info(f"Tool {tool_name} completed: {result_info}")
    
    async def before_resource_read(self, resource_uri: str) -> Optional[str]:
        self.request_count += 1
        logger.info(f"Resource read #{self.request_count}: {resource_uri}")
        return None
    
    async def after_resource_read(self, resource_uri: str, result: Any) -> None:
        result_info = self._get_result_info(result)
        logger.info(f"Resource {resource_uri} read: {result_info}")
    
    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from arguments."""
        safe_args = args.copy()
        # Truncate long query strings for logging
        if 'query' in safe_args and len(str(safe_args['query'])) > 100:
            safe_args['query'] = str(safe_args['query'])[:100] + "..."
        return safe_args
    
    def _get_result_info(self, result: Any) -> str:
        """Get safe information about the result for logging."""
        if isinstance(result, str):
            try:
                json_result = json.loads(result)
                if isinstance(json_result, dict):
                    if 'error' in json_result:
                        return f"error: {json_result.get('error_type', 'unknown')}"
                    elif 'papers' in json_result:
                        return f"{len(json_result.get('papers', []))} papers"
                    else:
                        return f"success ({len(str(result))} chars)"
            except:
                pass
            return f"string ({len(result)} chars)"
        return f"{type(result).__name__}"


# Authentication and rate limiting removed since arxiver doesn't need them currently


class SecurityMiddleware(MCPMiddleware):
    """Middleware for security checks and input sanitization."""
    
    def __init__(self):
        self.blocked_patterns = [
            r'<script',
            r'javascript:',
            r'sql.*injection',
            r'drop\s+table',
            r'delete\s+from',
        ]
        self.max_input_length = int(os.getenv('ARXIVER_MAX_INPUT_LENGTH', '10000'))
    
    async def before_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        # Check input length
        total_length = sum(len(str(v)) for v in args.values())
        if total_length > self.max_input_length:
            return create_error_response(
                f"Input too long: {total_length} chars (max: {self.max_input_length})",
                "security_error",
                {"tool": tool_name, "input_length": total_length}
            )
        
        # Check for malicious patterns
        for key, value in args.items():
            if isinstance(value, str):
                security_error = self._check_security_patterns(value, tool_name, key)
                if security_error:
                    return security_error
        
        return None
    
    def _check_security_patterns(self, input_text: str, tool_name: str, param_name: str) -> Optional[str]:
        """Check input for malicious patterns."""
        import re
        
        lower_input = input_text.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, lower_input, re.IGNORECASE):
                logger.warning(f"Blocked malicious input in {tool_name}.{param_name}: {pattern}")
                return create_error_response(
                    "Input contains potentially malicious content",
                    "security_error",
                    {"tool": tool_name, "parameter": param_name}
                )
        return None


# Initialize middleware instances (only logging and security for arxiver)
middleware_instances = [
    LoggingMiddleware(),
    SecurityMiddleware()
]

# Middleware Integration Functions
async def apply_middleware_before_tool(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Apply all middleware before tool execution. Return error string if any middleware blocks."""
    for middleware in middleware_instances:
        error = await middleware.before_tool_call(tool_name, args)
        if error:
            return error
    return None

async def apply_middleware_after_tool(tool_name: str, args: Dict[str, Any], result: Any) -> None:
    """Apply all middleware after tool execution."""
    for middleware in middleware_instances:
        try:
            await middleware.after_tool_call(tool_name, args, result)
        except Exception as e:
            logger.error(f"Middleware error in {middleware.__class__.__name__}: {e}")

async def apply_middleware_before_resource(resource_uri: str) -> Optional[str]:
    """Apply all middleware before resource read. Return error string if any middleware blocks."""
    for middleware in middleware_instances:
        error = await middleware.before_resource_read(resource_uri)
        if error:
            return error
    return None

async def apply_middleware_after_resource(resource_uri: str, result: Any) -> None:
    """Apply all middleware after resource read."""
    for middleware in middleware_instances:
        try:
            await middleware.after_resource_read(resource_uri, result)
        except Exception as e:
            logger.error(f"Middleware error in {middleware.__class__.__name__}: {e}")

# Initialize the FastMCP server
app = FastMCP("arxiver")


# Add resources for better paper browsing
@app.resource("arxiver://recent-papers")
async def list_recent_papers() -> str:
    """List recent papers from the database."""
    resource_uri = "arxiver://recent-papers"
    
    # Apply middleware before resource read
    middleware_error = await apply_middleware_before_resource(resource_uri)
    if middleware_error:
        return middleware_error
    
    try:
        conn = create_connection(PAPERS_DB)
        if not conn:
            result = "Database connection failed"
            await apply_middleware_after_resource(resource_uri, result)
            return result

        # Get recent papers (last 7 days by default)
        papers = get_recent_papers_since_days(conn, 7)
        conn.close()

        if not papers:
            result = "No recent papers found"
            await apply_middleware_after_resource(resource_uri, result)
            return result

        # Format as readable text
        result = f"Recent Papers ({len(papers)} found):\n\n"
        for i, paper in enumerate(papers[:20], 1):  # Limit to 20 papers
            # Convert Row to dict-like access
            title = paper["title"] if paper["title"] else "Unknown Title"
            authors = paper["authors"] if paper["authors"] else "Unknown"
            paper_id = paper["paper_id"] if paper["paper_id"] else "Unknown"
            categories = paper["categories"] if paper["categories"] else "Unknown"
            arxiv_url = (
                paper["arxiv_url"]
                if paper["arxiv_url"]
                else f"https://arxiv.org/abs/{paper_id}"
            )

            result += f"{i}. {title}\n"
            result += f"   Authors: {authors}\n"
            result += f"   arXiv ID: {paper_id}\n"
            result += f"   Categories: {categories}\n"
            result += f"   URL: {arxiv_url}\n\n"

        await apply_middleware_after_resource(resource_uri, result)
        return result

    except Exception as e:
        logger.error(f"Error in list_recent_papers resource: {e}")
        result = f"Error: {str(e)}"
        await apply_middleware_after_resource(resource_uri, result)
        return result


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


def get_cache_key(*args) -> str:
    """Generate cache key from arguments."""
    return "|".join(str(arg) for arg in args)


def get_from_cache(key: str) -> Optional[dict]:
    """Get value from cache if not expired."""
    if key not in _cache:
        return None

    timestamp = _cache_timestamps.get(key, 0)
    if datetime.now().timestamp() - timestamp > CACHE_TTL:
        # Cache expired, remove it
        _cache.pop(key, None)
        _cache_timestamps.pop(key, None)
        return None

    return _cache[key]


def set_cache(key: str, value: dict) -> None:
    """Set value in cache with current timestamp."""
    _cache[key] = value
    _cache_timestamps[key] = datetime.now().timestamp()


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


# Helper Functions for Enhanced Type Safety and Error Handling
def create_error_response(
    error_msg: str,
    error_type: str = "general",
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a standardized error response."""
    error_response = ErrorResponse(
        error=error_msg, error_type=error_type, details=details or {}
    )
    return error_response.model_dump_json()


def create_paper_from_row(row: Any) -> Paper:
    """Convert database row to Paper model."""
    try:
        return Paper(
            paper_id=str(row["paper_id"]) if row["paper_id"] else "",
            title=str(row["title"]) if row["title"] else "Unknown Title",
            authors=str(row["authors"]) if row["authors"] else None,
            published=str(row["published"]) if row["published"] else None,
            categories=str(row["categories"]) if row["categories"] else None,
            summary=str(row["summary"]) if row["summary"] else None,
            concise_summary=str(row["concise_summary"])
            if row["concise_summary"]
            else None,
            arxiv_url=f"https://arxiv.org/abs/{clean_paper_id(row['paper_id'])}"
            if row["paper_id"]
            else None,
        )
    except Exception as e:
        logger.error(f"Error creating Paper from row: {e}")
        # Return a minimal valid Paper object
        return Paper(
            paper_id=str(row.get("paper_id", "unknown")),
            title=str(row.get("title", "Unknown Title")),
        )


def validate_search_input(query: str, top_k: int) -> Optional[str]:
    """Validate search input parameters."""
    if not query or not query.strip():
        return "Query cannot be empty"
    if top_k <= 0:
        return "top_k must be greater than 0"
    if top_k > 100:
        return "top_k cannot exceed 100"
    return None


def measure_execution_time(start_time: float) -> float:
    """Calculate execution time in milliseconds."""
    import time

    return (time.time() - start_time) * 1000


# Tool definitions using FastMCP with Enhanced Type Safety
@app.tool()
async def search_papers(query: str, top_k: int = 5) -> str:
    """Search arXiv papers using semantic similarity based on embeddings.

    Args:
        query: Search query (e.g., 'machine learning transformers', 'computer vision')
        top_k: Number of papers to return (default: 5, max: 50)

    Returns:
        JSON string containing structured search results
    """
    import time

    start_time = time.time()
    args = {"query": query, "top_k": top_k}
    
    # Apply middleware before tool execution
    middleware_error = await apply_middleware_before_tool("search_papers", args)
    if middleware_error:
        return middleware_error

    # Enhanced input validation using helper function
    validation_error = validate_search_input(query, top_k)
    if validation_error:
        result = create_error_response(validation_error, "validation_error")
        await apply_middleware_after_tool("search_papers", args, result)
        return result

    try:
        impl_result = await search_papers_impl(query.strip(), top_k)

        # Convert to structured response
        papers = []
        if "papers" in impl_result and impl_result["papers"]:
            for paper_data in impl_result["papers"]:
                try:
                    paper = Paper(
                        paper_id=paper_data.get("paper_id", ""),
                        title=paper_data.get("title", "Unknown Title"),
                        authors=paper_data.get("authors"),
                        published=paper_data.get("published"),
                        categories=paper_data.get("categories"),
                        summary=paper_data.get("summary"),
                        concise_summary=paper_data.get("concise_summary"),
                        arxiv_url=paper_data.get("arxiv_url"),
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing paper data: {e}")
                    continue

        search_response = SearchResponse(
            query=query.strip(),
            total_results=impl_result.get("total_results", len(papers)),
            papers=papers,
            search_method=impl_result.get("search_method", "vector_similarity"),
            execution_time_ms=measure_execution_time(start_time),
        )

        result = search_response.model_dump_json(indent=2)
        await apply_middleware_after_tool("search_papers", args, result)
        return result

    except Exception as e:
        logger.error(f"Search papers error: {e}")
        result = create_error_response(
            f"Search operation failed: {str(e)}",
            "search_error",
            {"query": query, "top_k": top_k},
        )
        await apply_middleware_after_tool("search_papers", args, result)
        return result


@app.tool()
async def get_recommendations(days_back: int = 3) -> str:
    """Get personalized paper recommendations using ML model based on your interests.

    Args:
        days_back: Number of days to look back for recommendations (default: 3)

    Returns:
        JSON string containing recommended papers
    """
    # Input validation
    if not isinstance(days_back, int) or days_back < 1:
        return json.dumps(
            {
                "error": "Invalid input",
                "message": "days_back must be a positive integer",
            },
            indent=2,
        )

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
    # Input validation
    if not paper_id or not paper_id.strip():
        return json.dumps(
            {"error": "Invalid input", "message": "paper_id cannot be empty"}, indent=2
        )

    result = await summarize_paper_impl(paper_id.strip())
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
    # Input validation
    if not query or not query.strip():
        return json.dumps(
            {"error": "Invalid input", "message": "Query cannot be empty"}, indent=2
        )

    if not isinstance(top_i, int) or top_i < 1:
        return json.dumps(
            {"error": "Invalid input", "message": "top_i must be a positive integer"},
            indent=2,
        )

    if not isinstance(search_k, int) or search_k < 1:
        return json.dumps(
            {
                "error": "Invalid input",
                "message": "search_k must be a positive integer",
            },
            indent=2,
        )

    result = await choose_best_papers_impl(query.strip(), top_i, search_k)
    return json.dumps(result, indent=2)


@app.tool()
async def import_paper(arxiv_id: str) -> str:
    """Import a specific paper from arXiv into the database.

    Args:
        arxiv_id: arXiv ID without URL (e.g., '2404.04292' or '1706.03762')

    Returns:
        JSON string confirming import status
    """
    # Input validation
    if not arxiv_id or not arxiv_id.strip():
        return json.dumps(
            {"error": "Invalid input", "message": "arxiv_id cannot be empty"}, indent=2
        )

    result = await import_paper_impl(arxiv_id.strip())
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
    # Input validation
    if not paper_id or not paper_id.strip():
        return json.dumps(
            {"error": "Invalid input", "message": "paper_id cannot be empty"}, indent=2
        )

    result = await get_paper_details_impl(paper_id.strip())
    return json.dumps(result, indent=2)


@app.tool()
async def search_papers_advanced(
    query: str,
    top_k: int = 5,
    category_filter: str = "",
    date_from: str = "",
    date_to: str = "",
    min_score: float = 0.0,
) -> str:
    """Advanced search for arXiv papers with filtering options.

    Args:
        query: Search query text
        top_k: Number of papers to return (default: 5, max: 50)
        category_filter: Filter by arXiv category (e.g., 'cs.AI', 'cs.LG')
        date_from: Filter papers from this date (YYYY-MM-DD format)
        date_to: Filter papers up to this date (YYYY-MM-DD format)
        min_score: Minimum similarity score threshold (0.0-1.0)

    Returns:
        JSON string containing filtered search results
    """
    # Input validation
    if not query or not query.strip():
        return json.dumps(
            {"error": "Invalid input", "message": "Query cannot be empty"}, indent=2
        )

    if not isinstance(top_k, int) or top_k < 1:
        return json.dumps(
            {"error": "Invalid input", "message": "top_k must be a positive integer"},
            indent=2,
        )

    result = await search_papers_advanced_impl(
        query.strip(), top_k, category_filter, date_from, date_to, min_score
    )
    return json.dumps(result, indent=2)


@app.tool()
async def get_trending_papers(days_back: int = 7, category: str = "") -> str:
    """Get trending papers based on recent activity and popularity.

    Args:
        days_back: Number of days to look back (default: 7)
        category: Filter by specific category (optional)

    Returns:
        JSON string containing trending papers
    """
    # Input validation
    if not isinstance(days_back, int) or days_back < 1:
        return json.dumps(
            {
                "error": "Invalid input",
                "message": "days_back must be a positive integer",
            },
            indent=2,
        )

    result = await get_trending_papers_impl(days_back, category)
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

        # Check cache first
        cache_key = get_cache_key("search", query, top_k)
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached search results for '{query}'")
            return cached_result

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

        result = {"query": query, "total_results": len(papers), "papers": papers}

        # Cache the result
        set_cache(cache_key, result)

        return result

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


async def search_papers_advanced_impl(
    query: str,
    top_k: int = 5,
    category_filter: str = "",
    date_from: str = "",
    date_to: str = "",
    min_score: float = 0.0,
) -> dict:
    """Implementation for advanced search with filtering."""
    try:
        # First perform basic search
        basic_results = await search_papers_impl(
            query, min(top_k * 3, 150)
        )  # Get more results to filter

        if "error" in basic_results:
            return basic_results

        papers = basic_results.get("papers", [])

        # Apply filters
        filtered_papers = []

        for paper in papers:
            # Apply score filter
            distance = paper.get("distance", 1.0)
            similarity_score = 1.0 - distance if distance is not None else 0.0

            if similarity_score < min_score:
                continue

            # Apply category filter
            if category_filter:
                metadata = paper.get("metadata", {})
                paper_categories = metadata.get("categories", "")
                if category_filter.lower() not in paper_categories.lower():
                    continue

            # Apply date filters (if metadata contains date info)
            if date_from or date_to:
                metadata = paper.get("metadata", {})
                paper_date = metadata.get("published", "")

                if date_from and paper_date and paper_date < date_from:
                    continue
                if date_to and paper_date and paper_date > date_to:
                    continue

            # Add similarity score to paper info
            paper["similarity_score"] = similarity_score
            filtered_papers.append(paper)

        # Sort by similarity score and limit results
        filtered_papers.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        filtered_papers = filtered_papers[:top_k]

        return {
            "query": query,
            "filters": {
                "category": category_filter,
                "date_from": date_from,
                "date_to": date_to,
                "min_score": min_score,
            },
            "total_results": len(filtered_papers),
            "papers": filtered_papers,
        }

    except Exception as e:
        logger.error(f"Error in search_papers_advanced: {e}")
        return {"error": str(e), "message": "Error occurred during advanced search"}


async def get_trending_papers_impl(days_back: int = 7, category: str = "") -> dict:
    """Implementation for trending papers based on recent activity."""
    try:
        # Get recent papers from database
        conn = create_connection(PAPERS_DB)
        if not conn:
            return {
                "error": "Database connection failed",
                "message": "Could not connect to papers database",
            }

        # Build query with optional category filter
        query = """
            SELECT paper_id, title, authors, categories, published, arxiv_url, importance_score
            FROM papers 
            WHERE published >= date('now', '-{} days')
        """.format(days_back)

        params = []
        if category:
            query += " AND categories LIKE ?"
            params.append(f"%{category}%")

        query += " ORDER BY importance_score DESC, published DESC LIMIT 50"

        cursor = conn.cursor()
        cursor.execute(query, params)
        papers = cursor.fetchall()
        conn.close()

        if not papers:
            return {
                "message": f"No trending papers found in the last {days_back} days",
                "papers": [],
            }

        # Format results
        trending_papers = []
        for paper in papers:
            trending_papers.append(
                {
                    "paper_id": paper[0],
                    "title": paper[1],
                    "authors": paper[2] or "",
                    "categories": paper[3] or "",
                    "published": paper[4] or "",
                    "arxiv_url": paper[5] or f"https://arxiv.org/abs/{paper[0]}",
                    "importance_score": paper[6] or 0.0,
                    "trending_rank": len(trending_papers) + 1,
                }
            )

        return {
            "days_back": days_back,
            "category_filter": category,
            "total_papers": len(trending_papers),
            "trending_papers": trending_papers,
        }

    except Exception as e:
        logger.error(f"Error in get_trending_papers: {e}")
        return {
            "error": str(e),
            "message": "Error occurred while retrieving trending papers",
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
