import sqlite3
from datetime import datetime

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

INTERVAL_HOURS = 5  # Assuming you want to run this once a day

DATABASE_PATH = "../data/arxiv_papers.db"
SUMMARIZE_URL = "http://127.0.0.1:8000/summarize"


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn


def find_articles_to_summarize(conn, limit=10):
    """Find articles without concise_summary field"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT paper_id FROM papers WHERE concise_summary IS NULL OR concise_summary = '' LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def summarize_articles():
    print(f"Current time: {datetime.now()}")
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
            else:
                print(
                    f"  - Failed to summarize article {paper_id}: {response.status_code}, {response.text}"
                )
        conn.close()
    else:
        print("Failed to connect to the database.")


scheduler = BlockingScheduler()
scheduler.add_job(summarize_articles, "interval", minutes=INTERVAL_HOURS)

print(
    f"Starting scheduler to run /summarize every {INTERVAL_HOURS} hours for articles without concise summaries..."
)
summarize_articles()  # Run once immediately before starting the scheduler
scheduler.start()
