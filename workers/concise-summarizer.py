import sqlite3
from datetime import datetime

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

INTERVAL_MINUTES = 10

DATABASE_PATH = "../data/arxiv_papers.db"
SUMMARIZE_URL = "http://127.0.0.1:8000/summarize"


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn


def find_articles_to_summarize(conn, limit=100):
    """Find articles without concise_summary field"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT paper_id FROM papers WHERE concise_summary IS NULL OR concise_summary = '' LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def summarize_articles():
    print(f"Current time: {datetime.now()}")
    count = 0
    conn = create_connection(DATABASE_PATH)
    if conn is not None:
        articles = find_articles_to_summarize(conn)
        for article_id in articles:
            paper_id = article_id[0]
            print(f"- Processing article {paper_id}")
            response = requests.post(
                SUMMARIZE_URL,
                json={"paper_id": paper_id},
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                print(f"  - Successfully summarized article {paper_id}")
                count += 1
            else:
                print(
                    f"  - Failed to summarize article {paper_id}: {response.status_code}, {response.text}"
                )
        conn.close()
        print(f"-> Processed {count} new articles")
    else:
        print("Failed to connect to the database.")


scheduler = BlockingScheduler()
scheduler.add_job(summarize_articles, "interval", minutes=INTERVAL_MINUTES)

print(
    f"Starting scheduler to run /summarize every {INTERVAL_MINUTES} minutes for articles without concise summaries..."
)

# Run once before starting the scheduler
summarize_articles()
scheduler.start()
