import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest
import responses

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .arxiv import fetch_article_for_id, fetch_articles_for_date
except ImportError:
    from arxiv import fetch_article_for_id, fetch_articles_for_date


@responses.activate
def test_fetch_articles_for_date():
    from unittest.mock import patch

    with patch("arxiv.time.sleep", return_value=None):
        mock_conn = MagicMock()
        responses.add(
            responses.GET,
            "http://export.arxiv.org/api/query?search_query=(test)%20AND%20submittedDate:[202001010000%20TO%20202001012359]&start=0&max_results=100",
            body='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/test_id</id><title>Test Title</title><summary>Test Summary</summary><updated>2020-01-01T00:00:00Z</updated><published>2020-01-01T00:00:00Z</published></entry></feed>',
            status=200,
            content_type="application/atom+xml",
        )
        result = fetch_articles_for_date(
            mock_conn, "test", datetime.strptime("20200101", "%Y%m%d"), 10
        )
        assert len(result) > 0


@responses.activate
def test_fetch_article_for_id():
    from unittest.mock import patch

    with patch("arxiv.time.sleep", return_value=None):
        responses.add(
            responses.GET,
            "http://export.arxiv.org/api/query?id_list=1234.56789",
            body='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.56789v1</id><title>Specific Article</title><summary>Specific Summary</summary><updated>2020-01-01T00:00:00Z</updated><published>2020-01-01T00:00:00Z</published></entry></feed>',
            status=200,
            content_type="application/atom+xml",
        )
        result = fetch_article_for_id("1234.56789")
        assert result is not None
        assert result["paper_id"] == "http://arxiv.org/abs/1234.56789v1"
        assert result["title"] == "Specific Article"


if __name__ == "__main__":
    pytest.ingest(10)
