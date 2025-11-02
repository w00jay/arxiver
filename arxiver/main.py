import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Optional

import chromadb
import click
import numpy as np
import requests
import tensorflow as tf
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arxiv import fetch_article_for_id, fetch_articles_for_date
from chromadb.utils import embedding_functions
from database import (
    add_interested_db_column,
    create_connection,
    create_table,
    get_paper_by_id,
    get_recent_entries,
    get_recent_papers_since_days,
    update_concise_summary,
)
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.exceptions import HTTPException
from llm import choose_summaries, summarize_summary
from pydantic import BaseModel
from vector_db import get_embedding

# Remove all handlers associated with the root logger object
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
LOOK_BACK_DAYS = 3

# Get the directory containing this script and construct absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "predictor")
PAPERS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_papers.db")
EMBEDDINGS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_embeddings.chroma")

load_dotenv()
app = FastAPI()


class IngestRequest(BaseModel):
    start_date: Optional[str] = None
    days: Optional[int] = LOOK_BACK_DAYS


# curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"start_date": "2024-03-01", "days": 2}'
@app.post("/ingest", description="Ingest articles from arXiv for the past N days.")
async def ingest_articles(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest articles from arXiv for the past N days.
    """

    background_tasks.add_task(ingest_process, request.start_date, request.days)
    return {"message": f"Ingestion process started for the past {request.days} days."}


def summarize_recent_entries(database):
    conn = create_connection(database)
    if conn is not None:
        recent_entries = get_recent_entries(conn, 1500)

        fetch_count = 0
        for entry in recent_entries:
            paper_id, title, summary, concise_summary = entry

            if not concise_summary or concise_summary.strip() == "":
                logger.debug("-" * 40)
                logger.debug(f"Original summary for '{title}':\n{summary}\n")
                logger.info("Generating a new concise summary...\n")

                concise_summary = summarize_summary(summary)
                logger.info(f"Concise summary for '{title}':\n{concise_summary}\n")

                update_concise_summary(conn, paper_id, concise_summary)
                fetch_count += 1
        conn.close()

        logger.info(f"Generated {fetch_count} new concise summaries.")


def summarize_article(database, paper_id):
    conn = create_connection(database)
    if conn is not None:
        paper = get_paper_by_id(conn, paper_id)

        paper_id = paper[0]
        title = paper[1]
        summary = paper[2]
        concise_summary = paper[3]

        if not concise_summary or concise_summary.strip() == "":
            logger.debug("-" * 40)
            logger.debug(f"Original summary for '{title}':\n{summary}\n")
            logger.info("Generating a new concise summary...\n")

            concise_summary = summarize_summary(summary)
            logger.info(f"Concise summary for '{title}':\n{concise_summary}\n")

            update_concise_summary(conn, paper_id, concise_summary)
            logger.info(f"Generated a new concise summary.")
        else:
            logger.info(f"Concise summary already exists for {paper_id}.")

        conn.close()


def ingest_process(start_date, days=LOOK_BACK_DAYS):
    conn = create_connection(PAPERS_DB)
    if conn is not None:
        create_table(conn)

        search_query = (
            r'(all:"machine learning" OR all:"deep learning" OR all:"supervised learning" '
            r'OR all:"unsupervised learning" OR all:"reinforcement learning") OR '
            r'(all:"artificial intelligence" OR all:"natural language processing" OR all:"computer vision" '
            r'OR all:"robotics" OR all:"knowledge representation" OR all:"search algorithms") OR '
            r'(all:"large language models" OR all:"transformers" OR all:"GPT" OR all:"BERT" '
            r'OR all:"few-shot learning" OR all:"zero-shot learning") OR '
            r'(all:"data science" OR all:"ethics in AI" OR all:"AI in healthcare" OR all:"AI and society")'
        )

        # Fetch articles for the past N days from the start date and add them to the database
        new_article_ids = []
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        elif start_date is None:
            start_date = datetime.now()

        for day_offset in range(days):
            query_date = start_date - timedelta(days=day_offset)
            article_ids = fetch_articles_for_date(conn, search_query, query_date, 1500)

            # Generate concise summaries for articles that don't have them yet
            processed_count = 0
            skipped_count = 0

            for article_id in article_ids:
                paper = get_paper_by_id(conn, article_id)
                if paper and (
                    not paper[3] or paper[3].strip() == ""
                ):  # Check concise_summary field
                    summarize_article(PAPERS_DB, article_id)
                    processed_count += 1
                else:
                    logger.info(
                        f"Skipping {article_id} - concise summary already exists."
                    )
                    skipped_count += 1

            logger.info(
                f"Summary generation complete for {query_date.strftime('%Y-%m-%d')}: "
                f"Processed {processed_count} papers, Skipped {skipped_count} papers"
            )

        conn.close()
    else:
        logger.error("Error! cannot create the database connection.")


class EmbedRequest(BaseModel):
    days: Optional[int] = LOOK_BACK_DAYS


# curl -X POST http://127.0.0.1:8000/embed -H "Content-Type: application/json" -d '{"days": 2}'
@app.post("/embed")
async def create_embeddings(request: EmbedRequest, background_tasks: BackgroundTasks):
    """
    Generate embeddings for the past N days.
    """

    background_tasks.add_task(generate_and_store_embeddings, request.days)
    return {"message": f"Embedding process initiated for the past {request.days} days."}


def generate_and_store_embeddings(days: int):
    # SQLite summaries
    conn = sqlite3.connect(PAPERS_DB)
    cursor = conn.cursor()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    cursor.execute(
        """SELECT paper_id, concise_summary FROM papers
                      WHERE updated >= ? AND updated <= ?""",
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    papers = cursor.fetchall()

    logger.info(f"Found {len(papers)} papers to embed.")
    conn.close()

    # ChromaDB for vector storage
    vdb = chromadb.PersistentClient(path=EMBEDDINGS_DB)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Use get_collection for safety
    try:
        vectors = vdb.get_collection(name="arxiver")
    except:
        vectors = vdb.create_collection(
            name="arxiver",
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"},
        )

    count = 0
    if not papers:
        logger.info("No results to embed.")
        return

    for paper in papers:
        paper_id, summary = paper
        logger.debug(f"id: {paper_id}, summary: {summary}")

        if not summary:
            logger.info(f"Skipping {paper_id} as it has no summary.")
            continue

        # skip if the embedding already exists
        res = vectors.get(ids=[paper_id], limit=1)
        logger.debug(res["ids"])
        if paper_id in res["ids"]:
            logger.info(f"Embedding for {paper_id} already exists.")
            continue

        logger.info(f"Adding {paper_id}")
        vectors.add(
            documents=[summary], metadatas=[{"source": "arxiv"}], ids=[paper_id]
        )
        count += 1

    logger.info(f"Stored {count} new embeddings.")


class QueryRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 5


# curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query_text": "summary of latest machine learning"}'
@app.post("/query")
async def query_articles(request: QueryRequest):
    """
    Query the embeddings for the given text.
    Enhanced with better error handling.
    """

    try:
        vdb = chromadb.PersistentClient(path=EMBEDDINGS_DB)
        sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Use ChromaDB manager for concurrent-safe access
        from .chromadb_manager import chromadb_manager

        with chromadb_manager.get_collection_context(allow_concurrent=True) as vectors:
            # Check if collection has any documents
            collection_count = vectors.count()
            if collection_count == 0:
                logger.warning("ChromaDB collection is empty. No embeddings to query.")
                return {
                    "message": "No embeddings available. Please run /fill-missing-embeddings first.",
                    "results": [],
                }

            results = vectors.query(
                query_texts=[request.query_text],
                n_results=request.top_k if request.top_k else 5,
                include=["documents", "distances", "metadatas"],
            )

            res = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    item = {
                        "id": results["ids"][0][i],
                        "summary": results["documents"][0][i]
                        if results["documents"]
                        else "",
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else 0,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                    }
                    res.append(item)

            logger.info(f"Query '{request.query_text}' returned {len(res)} results")
            return res

    except Exception as e:
        logger.error(f"Error in query_articles: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


class ChooseRequest(BaseModel):
    query_text: str
    i: Optional[int] = 5
    k: Optional[int] = 50


"""
curl -X POST http://localhost:8000/choose \
  -H "Content-Type: application/json" \
  -d '{"query_text": "cutting edge latest technique on large language model", "i": 2, "k": 10}' \
| jq .
"""


@app.post("/choose")
def choose_process(
    request: ChooseRequest,
    description="Choose the top i summaries from the top k relevant articles.",
):
    """
    Choose the top i summaries from the top k relevant articles.
    """

    query_text = request.query_text
    i = request.i
    k = request.k

    logger.info(f"Querying for '{query_text[:20]}'...")
    base_url = "http://127.0.0.1:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {"query_text": query_text, "top_k": k}

    response = requests.post(base_url, json=data, headers=headers)
    results = response.json()

    logger.info(f"Choosing {i} summaries from {k} relevant articles...")
    response = choose_summaries(results, i)

    logger.debug(json.dumps(response, indent=2))

    # try:
    #     choices = json.loads(response)
    # except json.JSONDecodeError:
    #     logger.debug(response)
    #     raise HTTPException(
    #         status_code=500, detail="Error decoding the response from the model."
    #     )

    return response


class SummarizeRequest(BaseModel):
    paper_id: str


# curl -X POST http://127.0.0.1:8000/summarize -H "Content-Type: application/json" -d '{"paper_id": "http://arxiv.org/abs/2404.04292v1"}'
@app.post("/summarize")
async def create_concise_summary(request: SummarizeRequest):
    """
    Generate a concise summary for the given paper.
    """

    conn = create_connection(PAPERS_DB)
    if conn is not None:
        cursor = conn.cursor()

        # Fetch the specific entry
        cursor.execute(
            "SELECT paper_id, title, summary, concise_summary FROM papers WHERE paper_id = ?",
            (request.paper_id,),
        )
        entry = cursor.fetchone()

        if not entry:
            conn.close()
            raise HTTPException(status_code=404, detail="Paper not found")

        paper_id, title, summary, concise_summary = entry

        # Check if a concise summary already exists
        if concise_summary and concise_summary.strip() != "":
            conn.close()
            return {
                "message": "Concise summary already exists.",
                "concise_summary": concise_summary,
            }

        # Generate a new concise summary
        logger.debug(f"Original summary for '{title}':\n{summary}\n")
        concise_summary = summarize_summary(summary)
        logger.debug(f"Concise summary for '{title}':\n{concise_summary}\n")

        # Update the database with the new concise summary
        update_concise_summary(conn, paper_id, concise_summary)
        conn.close()

        return {
            "message": "Generated a new concise summary.",
            "concise_summary": concise_summary,
        }

    else:
        raise HTTPException(status_code=500, detail="Database connection error")


# curl -X GET http://127.0.0.1:8000/fill-missing-summaries
@app.get("/fill-missing-summaries")
async def fill_missing_summaries():
    """
    Fill missing concise summaries in the SQLite database for papers.
    """

    conn = sqlite3.connect(PAPERS_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT paper_id, summary FROM papers WHERE concise_summary IS NULL")
    papers = cursor.fetchall()
    logging.info(
        f"Found {len(papers)} papers with missing concise summaries in the SQLite database."
    )

    missing_summaries = []
    count = 0
    for paper_id, summary in papers:
        if summary is None:
            logging.info(f"Skipping {paper_id} as it has no summary.")
            continue

        summarize_article(PAPERS_DB, paper_id)

        logger.info(f"Article imported from {paper_id}")
        count += 1

    logging.info(f"Found {len(missing_summaries)} missing concise summaries.")
    logging.info(f"Added {count} missing concise summaries.")

    conn.close()

    return {"missing_summaries": missing_summaries}


# curl -X GET http://127.0.0.1:8000/fill-missing-embeddings
@app.get("/fill-missing-embeddings")
async def fill_missing_embeddings():
    """
    Fill missing embeddings in Chromadb for papers in the SQLite database.
    Enhanced version with batch processing and better error handling.
    """

    try:
        # Debug working directory and paths
        cwd = os.getcwd()
        logging.info(f"Working directory: {cwd}")
        logging.info(
            f"Database path: {PAPERS_DB} (exists: {os.path.exists(PAPERS_DB)})"
        )
        logging.info(
            f"ChromaDB path: {EMBEDDINGS_DB} (exists: {os.path.exists(EMBEDDINGS_DB)})"
        )

        conn = sqlite3.connect(PAPERS_DB)
        cursor = conn.cursor()

        # Get papers with summaries only (much more efficient)
        cursor.execute(
            "SELECT paper_id, concise_summary FROM papers WHERE concise_summary IS NOT NULL AND concise_summary != ''"
        )
        papers_with_summaries = cursor.fetchall()
        logging.info(
            f"Found {len(papers_with_summaries)} papers with summaries in the database."
        )

        # ChromaDB for vector storage with proper resource management
        try:
            from chromadb_manager import chromadb_manager

            # Health check first
            if not chromadb_manager.health_check():
                logging.warning("ChromaDB health check failed, attempting reset...")
                chromadb_manager.reset_connection()
                if not chromadb_manager.health_check():
                    raise Exception("ChromaDB is not accessible after reset")

            # Get collection using context manager for safe access
            with chromadb_manager.get_collection_context() as vectors:
                logging.info(f"Got collection with {vectors.count()} documents")

                # Process in batches for better performance
                batch_size = 100
                count = 0
                errors = 0
                missing_paper_ids = []

                for i in range(0, len(papers_with_summaries), batch_size):
                    batch = papers_with_summaries[i : i + batch_size]

                    # Check which papers need embeddings
                    batch_ids = [paper[0] for paper in batch]

                    try:
                        # Get existing embeddings for this batch
                        existing = vectors.get(ids=batch_ids)
                        existing_ids = (
                            set(existing["ids"]) if existing["ids"] else set()
                        )

                        # Find missing ones
                        to_add_docs = []
                        to_add_ids = []
                        to_add_metas = []

                        for paper_id, summary in batch:
                            if paper_id not in existing_ids and summary:
                                to_add_docs.append(summary)
                                to_add_ids.append(paper_id)
                                to_add_metas.append({"source": "arxiv"})
                                missing_paper_ids.append(paper_id)

                        # Batch add missing embeddings
                        if to_add_docs:
                            vectors.add(
                                documents=to_add_docs,
                                metadatas=to_add_metas,
                                ids=to_add_ids,
                            )
                            count += len(to_add_docs)
                            logging.info(
                                f"Added {len(to_add_docs)} embeddings in batch {i//batch_size + 1}"
                            )

                    except Exception as e:
                        logging.error(
                            f"Error processing batch {i//batch_size + 1}: {e}"
                        )
                        errors += 1
                        continue

                logging.info(
                    f"Embedding fill completed: {count} added, {errors} errors"
                )
                conn.close()

                return {
                    "total_papers_checked": len(papers_with_summaries),
                    "embeddings_added": count,
                    "errors": errors,
                    "missing_paper_ids": missing_paper_ids[:100],  # Limit response size
                }

        except Exception as e:
            conn.close()
            logging.error(f"ChromaDB connection failed: {e}")
            raise HTTPException(status_code=500, detail=f"ChromaDB error: {e}")

    except Exception as e:
        logging.error(f"Error in fill_missing_embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


class ImportRequest(BaseModel):
    arxiv_id: str  # arXiv article ID


# curl -X POST http://127.0.0.1:8000/import -H "Content-Type: application/json" -d '{"arxiv_id": "1706.03762"}'
@app.post("/import")
async def import_article(request: ImportRequest, background_tasks: BackgroundTasks):
    """
    Ingest a specific article from arXiv using the URL provided in paper_id.
    """

    background_tasks.add_task(import_process, request.arxiv_id)
    return {
        "message": f"Import process started for the article at URL: {request.arxiv_id}"
    }


def import_process(arxiv_id):
    logger.info(f"Starting import of article from {arxiv_id}")

    conn = create_connection(PAPERS_DB)

    paper_id = fetch_article_for_id(conn, arxiv_id)
    conn.close()

    summarize_article(PAPERS_DB, paper_id)

    logger.info(f"Article imported from {arxiv_id}")


def get_latest_model(directory):
    """Get the latest model file from a directory based on creation date."""
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".keras")
    ]
    if not files:
        return None
    logger.debug(files)
    latest_file = max(files, key=os.path.getctime)
    return latest_file


class RecommendRequest(BaseModel):
    days_back: int = LOOK_BACK_DAYS


# curl -X GET http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d '{"days_back": 1}' | jq .
@app.get("/recommend")
async def recommend(request: RecommendRequest):
    """
    Recommend papers based on the latest model.
    """

    # Find the latest keras model based on creation date
    latest_model_path = get_latest_model(MODEL_PATH)

    if not latest_model_path:
        raise HTTPException(
            status_code=404, detail="No model file found in the directory."
        )

    try:
        # Load the latest keras model
        model = tf.keras.models.load_model(latest_model_path, compile=False)
        logger.info(f"Loaded model: {latest_model_path}")

        # Get recent papers from the database
        conn = create_connection(PAPERS_DB)
        recent_papers = get_recent_papers_since_days(conn, days=request.days_back)
        logger.info(
            f"Got {len(recent_papers)} recent papers for {request.days_back} days."
        )

        # Reformat
        parsed_papers = []
        for paper in recent_papers:
            parsed = {
                "paper_id": paper[0],
                "title": paper[1].replace("\n", ""),
                "summary": paper[2],
                "concise_summary": paper[3],
            }
            parsed_papers.append(parsed)

        # Get the vector embeddings for the recent papers using batch query
        from arxiver.vector_db import get_embeddings_batch

        paper_ids = [p["paper_id"] for p in parsed_papers if p["paper_id"] is not None]
        logger.info(f"Fetching embeddings for {len(paper_ids)} papers in batch...")

        embeddings_dict = get_embeddings_batch(paper_ids)

        # Build new_X array from the batch results
        new_X = []
        embedding_errors = 0
        for paper in parsed_papers:
            if paper["paper_id"] is not None:
                embedding = embeddings_dict.get(paper["paper_id"])
                if embedding is not None:
                    new_X.append(embedding)
                else:
                    embedding_errors += 1

        if embedding_errors > 0:
            logger.warning(f"Failed to get embeddings for {embedding_errors} papers")

        if not new_X:
            logger.warning(
                f"No valid embeddings found in {len(recent_papers)} recent papers. "
                f"ChromaDB may need to be rebuilt. Returning fallback recommendations."
            )
            conn.close()

            # Return recent papers as fallback recommendations
            fallback_recommendations = []
            for paper in parsed_papers[:10]:  # Return top 10 recent papers
                fallback_recommendations.append(
                    {
                        "id": paper["paper_id"],
                        "title": paper["title"],
                        "summary": paper["concise_summary"]
                        or paper["summary"]
                        or "No summary available",
                    }
                )

            logger.info(
                f"Returning {len(fallback_recommendations)} fallback recommendations"
            )
            return fallback_recommendations

        # Convert list of embeddings to a 2D numpy array
        new_X = np.array(new_X)

        logger.info(
            f"Got {new_X.shape[0]} embeddings from {len(recent_papers)} recent papers. Making recommendations..."
        )

        # Make predictions (=article recommendations)
        recommended_papers = model.predict(new_X) > 0.5
        logger.debug("Prediction complete.")

        # Report the predictions
        formatted = []
        for i, is_recommended in enumerate(recommended_papers):
            if is_recommended:
                paper = parsed_papers[i]
                paper_id = paper["paper_id"]
                summary = paper["concise_summary"]
                title = paper["title"].replace("\n", "")

                logger.info(
                    f"Recommending {title}\n{paper_id}\n{is_recommended}\n{summary}"
                )
                formatted.append({"id": paper_id, "title": title, "summary": summary})

        logger.info(
            f"Got {len(formatted)} new recommendations from {len(recent_papers)} recent papers."
        )

        conn.close()
        return formatted

    except Exception as e:
        logger.error(f"Error in recommend function: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load the model or process recommendations: {str(e)}",
        )


# CLI commands
@click.group()
def cli():
    pass


@cli.command()
def webserver():
    """Starts the FastAPI web server."""
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)


@cli.command()
@click.option(
    "--days",
    default=LOOK_BACK_DAYS,
    help="Number of days to look back for new articles.",
)
@click.option(
    "--start-date",
    default=None,
    help="Start date for the ingestion process.",
)
def ingest(start_date, days):
    """
    Performs the ingestion process directly via CLI, without starting the web server.
    """
    logger.info(
        f"CLI Ingestion process started for the {days} days since {start_date}."
    )
    ingest_process(start_date, days)


# add a cli option to --add-interested-column
@cli.command()
def add_interested_column():
    """Add an 'interested' column to the papers table."""
    conn = create_connection(PAPERS_DB)
    add_interested_db_column(conn)
    conn.close()
    logger.info("Added 'interested' column to the papers table.")


if __name__ == "__main__":
    cli()
