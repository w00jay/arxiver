import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import chromadb
import click
import requests
import uvicorn
from arxiv import fetch_and_store_articles_for_date
from chromadb.utils import embedding_functions
from database import (
    add_interested_db_column,
    create_connection,
    create_table,
    get_recent_entries,
    update_concise_summary,
)
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.exceptions import HTTPException
from llm import choose_summaries, summarize_summary
from pydantic import BaseModel

LOOK_BACK_DAYS = 5

load_dotenv()
app = FastAPI()


class IngestRequest(BaseModel):
    days: Optional[int] = LOOK_BACK_DAYS

# curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"days": 5}'
@app.post("/ingest")
async def ingest_articles(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_process, request.days)
    return {"message": f"Ingestion process started for the past {request.days} days."}


def summarize_recent_entries(database):
    conn = create_connection(database)
    if conn is not None:
        recent_entries = get_recent_entries(conn, 1500)

        fetch_count = 0
        for entry in recent_entries:
            paper_id, title, summary, concise_summary = entry

            if not concise_summary:
                print("-" * 40)
                print(f"Original summary for '{title}':\n{summary}\n")
                print("Generating a new concise summary...\n")

                concise_summary = summarize_summary(summary)
                print(f"Concise summary for '{title}':\n{concise_summary}\n")

                update_concise_summary(conn, paper_id, concise_summary)
                fetch_count += 1
        conn.close()

        print(f"Generated {fetch_count} new concise summaries.")


def ingest_process(days=LOOK_BACK_DAYS):
    database = "../data/arxiv_papers.db"
    conn = create_connection(database)
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

        today = datetime.now()
        for day_offset in range(days):
            query_date = today - timedelta(days=day_offset)
            fetch_and_store_articles_for_date(conn, search_query, query_date, 1500)

        summarize_recent_entries(database)

        conn.close()
    else:
        print("Error! cannot create the database connection.")


class EmbedRequest(BaseModel):
    days: Optional[int] = LOOK_BACK_DAYS

# curl -X POST http://127.0.0.1:8000/embed -H "Content-Type: application/json" -d '{"days": 1}'
@app.post("/embed")
async def create_embeddings(request: EmbedRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_and_store_embeddings, request.days)
    return {"message": f"Embedding process initiated for the past {request.days} days."}


def generate_and_store_embeddings(days: int):
    # SQLite summaries
    conn = sqlite3.connect("../data/arxiv_papers.db")
    cursor = conn.cursor()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    cursor.execute(
        """SELECT paper_id, concise_summary FROM papers
                      WHERE updated >= ? AND updated <= ?""",
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    papers = cursor.fetchall()

    print(f"Found {len(papers)} papers to embed.")
    conn.close()

    # ChromaDB for vector storage
    vdb = chromadb.PersistentClient(path="../data/arxiv_embeddings.chroma")

    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     api_key="os.environ['HF_API_KEY']",
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    embedding_func = sentence_transformer_ef
    vectors = vdb.get_or_create_collection(
        name="arxiver", embedding_function=embedding_func
    )

    count = 0
    if not papers:
        print("No results to embed.")
        return

    for paper in papers:
        paper_id, summary = paper
        print(f"id: {paper_id}, summary: {summary}")

        if not summary:
            print(f"Skipping {paper_id} as it has no summary.")
            continue

        # skip if the embedding already exists
        res = vectors.get(ids=[paper_id], limit=1)
        print(res["ids"])
        if paper_id in res["ids"]:
            print(f"Embedding for {paper_id} already exists.")
            continue

        print(f"Adding {paper_id}")
        vectors.add(
            documents=[summary], metadatas=[{"source": "arxiv"}], ids=[paper_id]
        )
        count += 1

    print(f"Stored {count} new embeddings.")


class QueryRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 5

# curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query_text": "summary of latest machine learning"}'
@app.post("/query")
async def query_articles(request: QueryRequest):
    vdb = chromadb.PersistentClient(path="../data/arxiv_embeddings.chroma")
    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     api_key="os.environ['HF_API_KEY']",
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    # vectors = vdb.get_or_create_collection(name="arxiver", embedding_function=embedding_func)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    embedding_func = sentence_transformer_ef
    vectors = vdb.get_or_create_collection(
        name="arxiver", embedding_function=embedding_func
    )

    results = vectors.query(
        query_texts=[request.query_text],
        n_results=request.top_k if request.top_k else 5,
    )

    res = []
    for i in range(len(results["ids"][0])):
        item = {"id": results["ids"][0][i], "summary": results["documents"][0][i]}
        res.append(item)

    print(res)

    return res


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
def choose_process(request: ChooseRequest):
    query_text = request.query_text
    i = request.i
    k = request.k

    print(f"Querying for '{query_text[:20]}'...")
    base_url = "http://127.0.0.1:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {"query_text": query_text, "top_k": k}

    response = requests.post(base_url, json=data, headers=headers)
    results = response.json()

    print(f"Choosing {i} summaries from {k} relevant articles...")
    response = choose_summaries(results, i)

    for choice in response:
        print(f"ID: {choice['id']}\nSummary: {choice['summary']}\nReason: {choice.get('reason')}")
    # try:
    #     choices = json.loads(response)
    # except json.JSONDecodeError:
    #     print(response)
    #     raise HTTPException(
    #         status_code=500, detail="Error decoding the response from the model."
    #     )

    return response


class SummarizeRequest(BaseModel):
    paper_id: str

# curl -X POST http://127.0.0.1:8000/summarize -H "Content-Type: application/json" -d '{"paper_id": "http://arxiv.org/abs/2404.04292v1"}'
@app.post("/summarize")
async def create_concise_summary(request: SummarizeRequest):
    conn = create_connection("../data/arxiv_papers.db")
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
        if concise_summary:
            conn.close()
            return {
                "message": "Concise summary already exists.",
                "concise_summary": concise_summary,
            }

        # Generate a new concise summary
        print(f"Original summary for '{title}':\n{summary}\n")
        concise_summary = summarize_summary(
            summary
        )  # Assuming this is your summarization function
        print(f"Concise summary for '{title}':\n{concise_summary}\n")

        # Update the database with the new concise summary
        update_concise_summary(conn, paper_id, concise_summary)
        conn.close()

        return {
            "message": "Generated a new concise summary.",
            "concise_summary": concise_summary,
        }

    else:
        raise HTTPException(status_code=500, detail="Database connection error")


# CLI commands
@click.group()
def cli():
    pass


@cli.command()
def webserver():
    """Starts the FastAPI web server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", reload=True)


@cli.command()
@click.option(
    "--days",
    default=LOOK_BACK_DAYS,
    help="Number of days to look back for new articles.",
)
def ingest(days):
    """
    Performs the ingestion process directly via CLI, without starting the web server.
    """
    print(f"CLI Ingestion process started for the past {days} days.")
    ingest_process(days)


# add a cli option to --add-interested-column
@cli.command()
def add_interested_column():
    """Add an 'interested' column to the papers table."""
    conn = create_connection("../data/arxiv_papers.db")
    add_interested_db_column(conn)
    conn.close()
    print("Added 'interested' column to the papers table.")


if __name__ == "__main__":
    cli()
