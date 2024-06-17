import logging
import time
import xml.etree.ElementTree as ET

import requests
from database import insert_article

logging.basicConfig(level=logging.INFO)


def fetch_articles_for_date(conn, search_query, date, results_per_page=10):
    base_url = "http://export.arxiv.org/api/query?"
    formatted_date = date.strftime("%Y%m%d")
    query_url = f"{base_url}search_query=({search_query}) AND submittedDate:[{formatted_date}0000 TO {formatted_date}2359]&start=0&max_results={results_per_page}"
    response = requests.get(query_url)
    root = ET.fromstring(response.content)

    count = 0
    article_ids = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        updated = entry.find("{http://www.w3.org/2005/Atom}updated").text
        insert_article(conn, (paper_id, title, summary, updated))

        logging.info(f"{title} at {paper_id} on {updated}")
        count += 1
        article_ids.append(paper_id)

    logging.info(f"Found {count} articles on {date}")

    # Sleep for 3 seconds to avoid rate limiting
    time.sleep(3)

    return article_ids


def fetch_article_for_id(conn, arxiv_id):
    base_url = "http://export.arxiv.org/api/query?"
    query_url = f"{base_url}id_list={arxiv_id}"
    response = requests.get(query_url)
    root = ET.fromstring(response.content)

    count = 0
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        new_arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        updated = entry.find("{http://www.w3.org/2005/Atom}updated").text
        insert_article(conn, (new_arxiv_id, title, summary, updated))
        count += 1

        logging.info(f"{title} at {new_arxiv_id} on {updated} updated with\n{summary}")

    logging.info(f"Found {count} articles on {new_arxiv_id}")
    return new_arxiv_id
