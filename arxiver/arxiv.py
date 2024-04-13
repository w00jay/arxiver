import requests
import xml.etree.ElementTree as ET
from database import insert_article


def fetch_and_store_articles_for_date(conn, search_query, date, results_per_page=10):
    base_url = "http://export.arxiv.org/api/query?"
    formatted_date = date.strftime("%Y%m%d")
    query_url = f"{base_url}search_query=({search_query}) AND submittedDate:[{formatted_date}0000 TO {formatted_date}2359]&start=0&max_results={results_per_page}"
    response = requests.get(query_url)
    root = ET.fromstring(response.content)

    count = 0
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        updated = entry.find("{http://www.w3.org/2005/Atom}updated").text
        insert_article(conn, (paper_id, title, summary, updated))

        print(f"{title} at {paper_id} on {updated}")
        count += 1

    print(f"Found {count} articles on {date}")
