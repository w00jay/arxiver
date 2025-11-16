import logging
import time
import xml.etree.ElementTree as ET

import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

try:
    from database import insert_article
except ImportError:
    from .database import insert_article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arxiv_entry(entry):
    """Parse an arXiv API entry and extract all metadata."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    # Extract basic fields with None checks
    id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
    summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
    updated_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
    published_elem = entry.find("{http://www.w3.org/2005/Atom}published")

    if id_elem is None or id_elem.text is None:
        raise ValueError("Entry missing required field: id")
    if title_elem is None or title_elem.text is None:
        raise ValueError("Entry missing required field: title")
    if summary_elem is None or summary_elem.text is None:
        raise ValueError("Entry missing required field: summary")
    if updated_elem is None or updated_elem.text is None:
        raise ValueError("Entry missing required field: updated")
    if published_elem is None or published_elem.text is None:
        raise ValueError("Entry missing required field: published")

    paper_id = id_elem.text
    title = title_elem.text.strip()
    summary = summary_elem.text.strip()
    updated = updated_elem.text
    published = published_elem.text

    # Extract authors
    authors = []
    for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name_elem = author.find("{http://www.w3.org/2005/Atom}name")
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text)
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
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def fetch_articles_for_date(conn, search_query, date, max_results=1500):
    """
    Fetch articles for a specific date using pagination to avoid API errors.

    Args:
        conn: Database connection
        search_query: arXiv search query string
        date: datetime object for the date to query
        max_results: Maximum total results to fetch (will be paginated)

    Returns:
        List of article IDs that were fetched and inserted
    """
    base_url = "http://export.arxiv.org/api/query?"
    formatted_date = date.strftime("%Y%m%d")
    headers = {"Connection": "close"}

    # Use pagination with safe page size of 100 to avoid API timeouts/errors
    page_size = 100
    total_article_ids = []
    start_index = 0

    while start_index < max_results:
        query_url = f"{base_url}search_query=({search_query}) AND submittedDate:[{formatted_date}0000 TO {formatted_date}2359]&start={start_index}&max_results={page_size}"

        # Ensure minimum 3 second delay before each request per arXiv requirements
        if start_index > 0:
            time.sleep(3)

        try:
            response = requests.get(query_url, headers=headers, timeout=(90, 180))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for date {date}, start={start_index}: {e}")
            raise

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logger.error(
                f"XML parsing failed for date {date}. Response length: {len(response.content)} bytes"
            )
            logger.debug(
                f"Response content (first 500 chars): {response.content[:500]}"
            )
            raise

        # Parse entries from this page
        page_count = 0
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            try:
                article_data = parse_arxiv_entry(entry)
                insert_article(conn, article_data)

                logging.info(
                    f"{article_data['title']} at {article_data['paper_id']} on {article_data['updated']}"
                )
                page_count += 1
                total_article_ids.append(article_data["paper_id"])
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse entry for date {date}: {e}")
                logger.debug(
                    f"Entry XML: {ET.tostring(entry, encoding='unicode')[:500]}"
                )
                continue

        # If we got fewer results than page_size, we've reached the end
        if page_count < page_size:
            logger.info(f"Reached end of results at page starting at {start_index}")
            break

        start_index += page_size

    logging.info(f"Found {len(total_article_ids)} articles on {date}")

    return total_article_ids


@retry(
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def fetch_article_for_id(arxiv_id):
    """Fetch a single article by its arXiv ID and return parsed data."""
    base_url = "http://export.arxiv.org/api/query?"
    query_url = f"{base_url}id_list={arxiv_id}"

    # Add 3 second delay to comply with arXiv rate limits
    time.sleep(3)

    response = requests.get(query_url, timeout=(90, 180))
    root = ET.fromstring(response.content)

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        article_data = parse_arxiv_entry(entry)
        logging.info(
            f"{article_data['title']} at {article_data['paper_id']} on {article_data['updated']}"
        )
        return article_data

    return None
