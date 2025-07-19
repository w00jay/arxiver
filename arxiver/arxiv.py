import logging
import time
import xml.etree.ElementTree as ET

import requests

try:
    from .database import insert_article
except ImportError:
    from database import insert_article
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arxiv_entry(entry):
    """Parse an arXiv API entry and extract all metadata."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    # Extract basic fields
    paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
    title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
    summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
    updated = entry.find("{http://www.w3.org/2005/Atom}updated").text
    published = entry.find("{http://www.w3.org/2005/Atom}published").text

    # Extract authors
    authors = []
    for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name = author.find("{http://www.w3.org/2005/Atom}name").text
        if name:
            authors.append(name)
    authors_str = ", ".join(authors)

    # Extract categories
    categories = []
    for category in entry.findall("{http://www.w3.org/2005/Atom}category"):
        term = category.get("term")
        if term:
            categories.append(term)
    categories_str = ", ".join(categories)

    # Extract links
    arxiv_url = paper_id  # Default to paper_id
    pdf_url = None
    for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
        if link.get("type") == "text/html":
            arxiv_url = link.get("href")
        elif link.get("type") == "application/pdf":
            pdf_url = link.get("href")

    # Keep paper_id as URL to match existing database format

    return {
        "paper_id": paper_id,
        "title": title,
        "summary": summary,
        "updated": updated,
        "published": published,
        "authors": authors_str,
        "categories": categories_str,
        "arxiv_url": arxiv_url,
        "pdf_url": pdf_url,
        "interested": 0,
    }


@retry(
    wait=wait_exponential(multiplier=1, min=6),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def fetch_articles_for_date(conn, search_query, date, results_per_page=100):
    base_url = "http://export.arxiv.org/api/query?"
    formatted_date = date.strftime("%Y%m%d")
    query_url = f"{base_url}search_query=({search_query}) AND submittedDate:[{formatted_date}0000 TO {formatted_date}2359]&start=0&max_results={results_per_page}"
    headers = {"Connection": "close"}
    response = requests.get(query_url, headers=headers, timeout=(90, 180))

    root = ET.fromstring(response.content)

    count = 0
    article_ids = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        article_data = parse_arxiv_entry(entry)
        insert_article(conn, article_data)

        logging.info(
            f"{article_data['title']} at {article_data['paper_id']} on {article_data['updated']}"
        )
        count += 1
        article_ids.append(article_data["paper_id"])

    logging.info(f"Found {count} articles on {date}")

    # Sleep for 3 seconds to avoid rate limiting
    time.sleep(3)

    return article_ids


@retry(
    wait=wait_exponential(multiplier=1, min=5),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def fetch_article_for_id(arxiv_id):
    """Fetch a single article by its arXiv ID and return parsed data."""
    base_url = "http://export.arxiv.org/api/query?"
    query_url = f"{base_url}id_list={arxiv_id}"
    response = requests.get(query_url)
    root = ET.fromstring(response.content)

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        article_data = parse_arxiv_entry(entry)
        logging.info(
            f"{article_data['title']} at {article_data['paper_id']} on {article_data['updated']}"
        )
        return article_data

    return None
