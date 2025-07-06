from datetime import datetime
from unittest.mock import MagicMock

import pytest
import responses

# Assuming these are defined in your module
from arxiv import fetch_article_for_id, fetch_articles_for_date


@responses.activate
def test_fetch_articles_for_date():
    mock_conn = MagicMock()
    responses.add(
        responses.GET,
        "http://export.arxiv.org/api/query?search_query=(test)%20AND%20submittedDate:[202001010000%20TO%20202001012359]&start=0&max_results=10",
        body='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>test_id</id><title>Test Title</title><summary>Test Summary</summary><updated>2020-01-01</updated></entry></feed>',
        status=200,
        content_type="application/atom+xml",
    )
    fetch_articles_for_date(
        mock_conn, "test", datetime.strptime("20200101", "%Y%m%d"), 10
    )
    mock_conn.insert_article.assert_called_once()


@responses.activate
def test_fetch_article_for_id():
    mock_conn = MagicMock()
    responses.add(
        responses.GET,
        "http://export.arxiv.org/api/query?id_list=1234.56789",
        body='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>1234.56789</id><title>Specific Article</title><summary>Specific Summary</summary><updated>2020-01-01</updated></entry></feed>',
        status=200,
        content_type="application/atom+xml",
    )
    result_id = fetch_article_for_id(mock_conn, "1234.56789")
    assert result_id == "1234.56789"
    mock_conn.insert_article.assert_called_once()


if __name__ == "__main__":
    pytest.ingest(10)
