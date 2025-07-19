#!/usr/bin/env python3
"""
Critical Ingestion Workflow Tests for Arxiver Application

These tests validate the complete ingestion process from arXiv API
to database storage, ensuring data integrity throughout.
"""

import os
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Add arxiver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arxiver"))

from arxiv import fetch_articles_for_date, parse_arxiv_entry
from database import (
    create_connection,
    create_table,
    get_paper_by_id,
    get_recent_entries,
)
from main import ingest_process, summarize_article


class TestIngestionWorkflow:
    """Test the complete ingestion workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        conn = create_connection(db_path)
        create_table(conn)
        conn.close()

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def create_mock_arxiv_response(self, papers: list) -> str:
        """Create a mock arXiv API response with multiple papers."""
        entries = []
        for paper in papers:
            entry = f"""
                <entry>
                    <id>{paper.get('id', 'http://arxiv.org/abs/2507.12345v1')}</id>
                    <title>{paper.get('title', 'Test Paper Title')}</title>
                    <summary>{paper.get('summary', 'Test summary content')}</summary>
                    <updated>{paper.get('updated', '2025-07-17T10:30:00Z')}</updated>
                    <published>{paper.get('published', '2025-07-17T09:00:00Z')}</published>
                    <author><name>{paper.get('author1', 'John Doe')}</name></author>
                    <author><name>{paper.get('author2', 'Jane Smith')}</name></author>
                    <category term="{paper.get('category1', 'cs.AI')}" />
                    <category term="{paper.get('category2', 'cs.LG')}" />
                    <link href="{paper.get('id', 'http://arxiv.org/abs/2507.12345v1')}" type="text/html" />
                    <link href="{paper.get('id', 'http://arxiv.org/abs/2507.12345v1')}.pdf" type="application/pdf" />
                </entry>
            """
            entries.append(entry)

        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            {''.join(entries)}
        </feed>"""

    @patch("arxiv.requests.get")
    def test_fetch_articles_for_date_success(self, mock_get, temp_db):
        """Test successful fetching and parsing of articles for a specific date."""
        # Mock arXiv API response
        test_papers = [
            {
                "id": "http://arxiv.org/abs/2507.12345v1",
                "title": "First Test Paper",
                "summary": "First paper summary",
                "author1": "Alice",
                "author2": "Bob",
            },
            {
                "id": "http://arxiv.org/abs/2507.12346v1",
                "title": "Second Test Paper",
                "summary": "Second paper summary",
                "author1": "Charlie",
                "author2": "Diana",
            },
        ]

        mock_response = MagicMock()
        mock_response.content = self.create_mock_arxiv_response(test_papers).encode(
            "utf-8"
        )
        mock_get.return_value = mock_response

        # Test fetch_articles_for_date
        conn = create_connection(temp_db)
        search_query = "test query"
        test_date = datetime(2025, 7, 17)

        article_ids = fetch_articles_for_date(conn, search_query, test_date, 10)

        # Verify results
        assert len(article_ids) == 2, "Should return 2 article IDs"
        assert "2507.12345v1" in article_ids
        assert "2507.12346v1" in article_ids

        # Verify papers were inserted into database
        paper1 = get_paper_by_id(conn, "2507.12345v1")
        paper2 = get_paper_by_id(conn, "2507.12346v1")

        assert paper1 is not None, "First paper should be in database"
        assert paper2 is not None, "Second paper should be in database"

        # Verify metadata was extracted correctly
        assert "First Test Paper" in paper1[1]  # title
        assert "First paper summary" in paper1[2]  # summary

        conn.close()

    @patch("arxiv.requests.get")
    def test_duplicate_ingestion_preserves_data(self, mock_get, temp_db):
        """Test that re-ingesting the same papers preserves existing data."""
        # First ingestion
        test_paper = {
            "id": "http://arxiv.org/abs/2507.12345v1",
            "title": "Test Paper",
            "summary": "Test summary",
        }

        mock_response = MagicMock()
        mock_response.content = self.create_mock_arxiv_response([test_paper]).encode(
            "utf-8"
        )
        mock_get.return_value = mock_response

        conn = create_connection(temp_db)

        # First ingestion
        article_ids = fetch_articles_for_date(conn, "test", datetime(2025, 7, 17), 10)
        assert len(article_ids) == 1

        # Add concise summary to simulate processing
        from database import update_concise_summary

        update_concise_summary(conn, "2507.12345v1", "Original concise summary")

        # Verify concise summary exists
        paper_before = get_paper_by_id(conn, "2507.12345v1")
        assert paper_before[3] == "Original concise summary"

        # Second ingestion (duplicate)
        article_ids_2 = fetch_articles_for_date(conn, "test", datetime(2025, 7, 17), 10)

        # Verify concise summary is preserved
        paper_after = get_paper_by_id(conn, "2507.12345v1")
        assert (
            paper_after[3] == "Original concise summary"
        ), "Concise summary should be preserved"

        conn.close()

    @patch("main.summarize_summary")
    def test_summarize_article_skip_logic(self, mock_summarize, temp_db):
        """Test that summarize_article skips papers that already have summaries."""
        # Insert a paper with existing concise summary
        conn = create_connection(temp_db)

        from database import insert_article, update_concise_summary

        test_article = {
            "paper_id": "2507.12345v1",
            "title": "Test Paper",
            "summary": "Test summary",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "Test Author",
            "published": "2025-07-17",
            "categories": "cs.AI",
            "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
            "interested": 0,
        }

        insert_article(conn, test_article)
        update_concise_summary(conn, "2507.12345v1", "Existing summary")
        conn.close()

        # Test summarize_article - should skip
        summarize_article(temp_db, "2507.12345v1")

        # Should not have called LLM
        mock_summarize.assert_not_called()

        # Test with paper that doesn't have summary
        conn = create_connection(temp_db)
        test_article_2 = {
            "paper_id": "2507.12346v1",
            "title": "Test Paper 2",
            "summary": "Test summary 2",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "Test Author 2",
            "published": "2025-07-17",
            "categories": "cs.AI",
            "arxiv_url": "http://arxiv.org/abs/2507.12346v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12346v1.pdf",
            "interested": 0,
        }
        insert_article(conn, test_article_2)
        conn.close()

        mock_summarize.return_value = "Generated summary"

        # Should call LLM for paper without summary
        summarize_article(temp_db, "2507.12346v1")
        mock_summarize.assert_called_once()

    @patch("arxiv.requests.get")
    @patch("main.summarize_summary")
    def test_complete_ingest_process_workflow(self, mock_summarize, mock_get, temp_db):
        """Test the complete ingest_process workflow."""
        # Mock arXiv response
        test_papers = [
            {
                "id": "http://arxiv.org/abs/2507.12345v1",
                "title": "ML Paper",
                "summary": "Machine learning research",
                "author1": "Researcher A",
                "author2": "Researcher B",
            }
        ]

        mock_response = MagicMock()
        mock_response.content = self.create_mock_arxiv_response(test_papers).encode(
            "utf-8"
        )
        mock_get.return_value = mock_response

        mock_summarize.return_value = "Generated concise summary"

        # Run complete ingest process
        start_date = "2025-07-17"
        days = 1

        # Temporarily patch PAPERS_DB path for testing
        with patch("main.PAPERS_DB", temp_db):
            ingest_process(start_date, days)

        # Verify paper was ingested
        conn = create_connection(temp_db)
        paper = get_paper_by_id(conn, "2507.12345v1")

        assert paper is not None, "Paper should be ingested"
        assert "ML Paper" in paper[1], "Title should be correct"
        assert "Machine learning research" in paper[2], "Summary should be correct"

        conn.close()

        # Verify LLM was called for summary generation
        mock_summarize.assert_called_once()

    def test_ingest_process_counts_logging(self, temp_db, caplog):
        """Test that ingest_process logs correct skip/process counts."""
        # Pre-populate database with papers
        conn = create_connection(temp_db)

        from database import insert_article, update_concise_summary

        # Paper with existing summary (should be skipped)
        existing_paper = {
            "paper_id": "2507.12345v1",
            "title": "Existing Paper",
            "summary": "Existing summary",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "Existing Author",
            "published": "2025-07-17",
            "categories": "cs.AI",
            "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
            "interested": 0,
        }
        insert_article(conn, existing_paper)
        update_concise_summary(conn, "2507.12345v1", "Existing concise summary")

        # Paper without summary (should be processed)
        new_paper = {
            "paper_id": "2507.12346v1",
            "title": "New Paper",
            "summary": "New summary",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "New Author",
            "published": "2025-07-17",
            "categories": "cs.LG",
            "arxiv_url": "http://arxiv.org/abs/2507.12346v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12346v1.pdf",
            "interested": 0,
        }
        insert_article(conn, new_paper)

        conn.close()

        # Mock fetch_articles_for_date to return these papers
        with patch("main.fetch_articles_for_date") as mock_fetch:
            mock_fetch.return_value = ["2507.12345v1", "2507.12346v1"]

            with patch("main.summarize_article") as mock_summarize:
                with patch("main.PAPERS_DB", temp_db):
                    ingest_process("2025-07-17", 1)

        # Check that logging includes correct counts
        assert "Processed 1 papers, Skipped 1 papers" in caplog.text

        # Verify summarize_article was called only once (for new paper)
        mock_summarize.assert_called_once_with(temp_db, "2507.12346v1")


class TestIngestionErrorHandling:
    """Test error handling in ingestion workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        conn = create_connection(db_path)
        create_table(conn)
        conn.close()

        yield db_path

        if os.path.exists(db_path):
            os.unlink(db_path)

    @patch("arxiv.requests.get")
    def test_handle_arxiv_api_error(self, mock_get, temp_db):
        """Test handling of arXiv API errors."""
        # Mock API failure
        mock_get.side_effect = Exception("API connection failed")

        conn = create_connection(temp_db)

        # Should handle error gracefully
        with pytest.raises(Exception):
            fetch_articles_for_date(conn, "test", datetime(2025, 7, 17), 10)

        conn.close()

    @patch("arxiv.requests.get")
    def test_handle_malformed_xml(self, mock_get, temp_db):
        """Test handling of malformed XML responses."""
        # Mock malformed XML response
        mock_response = MagicMock()
        mock_response.content = b"<invalid>malformed xml"
        mock_get.return_value = mock_response

        conn = create_connection(temp_db)

        # Should handle XML parsing error
        with pytest.raises(ET.ParseError):
            fetch_articles_for_date(conn, "test", datetime(2025, 7, 17), 10)

        conn.close()

    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        # Try to connect to non-existent database location
        invalid_path = "/invalid/path/database.db"

        # Should handle gracefully or raise appropriate error
        conn = create_connection(invalid_path)
        # This may or may not fail depending on implementation
        if conn:
            conn.close()


if __name__ == "__main__":
    # Run ingestion workflow tests
    import subprocess

    print("🔄 Running Ingestion Workflow Tests...")

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    exit_code = result.returncode
    if exit_code == 0:
        print("✅ All ingestion workflow tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(exit_code)
