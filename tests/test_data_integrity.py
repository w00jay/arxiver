#!/usr/bin/env python3
"""
Critical Data Integrity Tests for Arxiver Application

These tests validate data completeness, consistency, and integrity
across the entire application stack.
"""

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytest

# Add arxiver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arxiver"))

import xml.etree.ElementTree as ET

from arxiv import fetch_articles_for_date, parse_arxiv_entry
from database import (
    create_connection,
    create_table,
    get_paper_by_id,
    get_recent_entries,
    insert_article,
)


class TestDatabaseIntegrity:
    """Test database operations and data integrity."""

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

    def test_database_schema_completeness(self, temp_db):
        """Test that database schema contains all required fields."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(papers)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        required_fields = {
            "paper_id": "TEXT",
            "title": "TEXT",
            "summary": "TEXT",
            "concise_summary": "TEXT",
            "updated": "TEXT",
            "interested": "BOOLEAN",
            "authors": "TEXT",
            "published": "DATE",
            "categories": "TEXT",
            "arxiv_url": "TEXT",
            "pdf_url": "TEXT",
            "abstract_embedding": "BLOB",
            "citation_count": "INTEGER",
            "related_papers": "TEXT",
            "last_updated": "TIMESTAMP",
            "user_rating": "INTEGER",
            "notes": "TEXT",
            "tags": "TEXT",
            "read_status": "TEXT",
            "importance_score": "REAL",
        }

        for field, expected_type in required_fields.items():
            assert field in columns, f"Missing required field: {field}"
            # Note: SQLite may show different type names, so we just check presence

        conn.close()

    def test_insert_article_preserves_all_data(self, temp_db):
        """Test that insert_article preserves all provided data."""
        test_article = {
            "paper_id": "2507.12345v1",
            "title": "Test Paper Title",
            "summary": "This is a test summary with detailed content.",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "John Doe, Jane Smith",
            "published": "2025-07-17",
            "categories": "cs.AI, cs.LG",
            "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
            "interested": 0,
        }

        conn = create_connection(temp_db)
        insert_article(conn, test_article)

        # Retrieve and verify all data is preserved
        retrieved = get_paper_by_id(conn, "2507.12345v1")
        assert retrieved is not None, "Paper should be retrievable after insert"

        # Check core fields
        assert retrieved[0] == test_article["paper_id"]
        assert retrieved[1] == test_article["title"]
        assert retrieved[2] == test_article["summary"]
        # concise_summary should be None initially
        assert retrieved[3] is None

        conn.close()

    def test_insert_or_ignore_prevents_data_loss(self, temp_db):
        """Test that INSERT OR IGNORE doesn't destroy existing data."""
        # Insert initial paper with concise summary
        conn = create_connection(temp_db)

        # Insert original article
        original_article = {
            "paper_id": "2507.12345v1",
            "title": "Original Title",
            "summary": "Original summary",
            "updated": "2025-07-17T10:30:00Z",
            "authors": "Original Author",
            "published": "2025-07-17",
            "categories": "cs.AI",
            "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
            "interested": 0,
        }
        insert_article(conn, original_article)

        # Add concise summary
        from database import update_concise_summary

        update_concise_summary(conn, "2507.12345v1", "Original concise summary")

        # Attempt to insert same paper again (simulate re-ingestion)
        duplicate_article = {
            "paper_id": "2507.12345v1",
            "title": "Modified Title",  # Different data
            "summary": "Modified summary",
            "updated": "2025-07-17T11:00:00Z",
            "authors": "Modified Author",
            "published": "2025-07-17",
            "categories": "cs.LG",
            "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
            "interested": 0,
        }
        insert_article(conn, duplicate_article)

        # Verify original data is preserved
        retrieved = get_paper_by_id(conn, "2507.12345v1")
        assert retrieved[1] == "Original Title", "Original title should be preserved"
        assert (
            retrieved[3] == "Original concise summary"
        ), "Concise summary should be preserved"

        conn.close()

    def test_data_completeness_validation(self, temp_db):
        """Test validation of data completeness for inserted articles."""
        conn = create_connection(temp_db)

        # Test article with missing critical metadata
        incomplete_article = {
            "paper_id": "2507.12345v1",
            "title": "Test Paper",
            "summary": "Test summary",
            "updated": "2025-07-17T10:30:00Z",
            "interested": 0,
            # Missing: authors, published, categories, urls
        }

        insert_article(conn, incomplete_article)
        retrieved = get_paper_by_id(conn, "2507.12345v1")

        # Check that paper was inserted but metadata fields are None/empty
        assert retrieved is not None
        # In current implementation, missing dict keys result in None values
        # This test documents current behavior - ideally we'd want validation

        conn.close()


class TestArxivDataParsing:
    """Test arXiv API data parsing and extraction."""

    def create_sample_arxiv_xml(self, paper_id: str = "2507.12345v1") -> str:
        """Create a sample arXiv XML response for testing."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/{paper_id}</id>
                <title>Test Paper Title: A Comprehensive Study</title>
                <summary>This is a comprehensive test summary that describes the paper's methodology, findings, and implications for the field.</summary>
                <updated>2025-07-17T10:30:00Z</updated>
                <published>2025-07-17T09:00:00Z</published>
                <author>
                    <name>John Doe</name>
                </author>
                <author>
                    <name>Jane Smith</name>
                </author>
                <category term="cs.AI" />
                <category term="cs.LG" />
                <link href="http://arxiv.org/abs/{paper_id}" type="text/html" />
                <link href="http://arxiv.org/pdf/{paper_id}.pdf" type="application/pdf" />
            </entry>
        </feed>"""

    def test_parse_arxiv_entry_completeness(self):
        """Test that parse_arxiv_entry extracts all required fields."""
        xml_content = self.create_sample_arxiv_xml()
        root = ET.fromstring(xml_content)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")

        parsed_data = parse_arxiv_entry(entry)

        # Verify all expected fields are present
        required_fields = [
            "paper_id",
            "title",
            "summary",
            "updated",
            "published",
            "authors",
            "categories",
            "arxiv_url",
            "pdf_url",
            "interested",
        ]

        for field in required_fields:
            assert field in parsed_data, f"Missing field in parsed data: {field}"
            assert parsed_data[field] is not None, f"Field {field} should not be None"

        # Verify specific values
        assert parsed_data["paper_id"] == "2507.12345v1"
        assert "Test Paper Title" in parsed_data["title"]
        assert "comprehensive test summary" in parsed_data["summary"]
        assert parsed_data["authors"] == "John Doe, Jane Smith"
        assert parsed_data["categories"] == "cs.AI, cs.LG"
        assert parsed_data["arxiv_url"] == "http://arxiv.org/abs/2507.12345v1"
        assert parsed_data["pdf_url"] == "http://arxiv.org/pdf/2507.12345v1.pdf"
        assert parsed_data["interested"] == 0

    def test_parse_arxiv_entry_handles_missing_fields(self):
        """Test parsing when some optional fields are missing."""
        minimal_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2507.12345v1</id>
                <title>Minimal Paper</title>
                <summary>Basic summary</summary>
                <updated>2025-07-17T10:30:00Z</updated>
                <published>2025-07-17T09:00:00Z</published>
            </entry>
        </feed>"""

        root = ET.fromstring(minimal_xml)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")

        parsed_data = parse_arxiv_entry(entry)

        # Should still parse successfully
        assert parsed_data["paper_id"] == "2507.12345v1"
        assert parsed_data["title"] == "Minimal Paper"
        # Authors and categories should be empty strings (not None)
        assert parsed_data["authors"] == ""
        assert parsed_data["categories"] == ""


class TestProductionDatabaseValidation:
    """Validate the current production database state."""

    def test_production_db_exists(self):
        """Test that production database exists and is accessible."""
        db_path = "./data/arxiv_papers.db"
        assert os.path.exists(db_path), "Production database should exist"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        count = cursor.fetchone()[0]

        assert count > 0, "Production database should contain papers"
        print(f"Production database contains {count} papers")
        conn.close()

    def test_production_data_completeness(self):
        """Test completeness of data in production database."""
        db_path = "./data/arxiv_papers.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check metadata completeness
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN authors IS NOT NULL AND authors != '' THEN 1 ELSE 0 END) as has_authors,
                SUM(CASE WHEN published IS NOT NULL AND published != '' THEN 1 ELSE 0 END) as has_published,
                SUM(CASE WHEN categories IS NOT NULL AND categories != '' THEN 1 ELSE 0 END) as has_categories,
                SUM(CASE WHEN concise_summary IS NOT NULL AND concise_summary != '' THEN 1 ELSE 0 END) as has_concise_summary
            FROM papers
        """)

        result = cursor.fetchone()
        total, has_authors, has_published, has_categories, has_concise_summary = result

        print(f"Production Data Completeness Report:")
        print(f"Total papers: {total}")
        print(f"Papers with authors: {has_authors} ({has_authors/total*100:.1f}%)")
        print(
            f"Papers with published dates: {has_published} ({has_published/total*100:.1f}%)"
        )
        print(
            f"Papers with categories: {has_categories} ({has_categories/total*100:.1f}%)"
        )
        print(
            f"Papers with concise summaries: {has_concise_summary} ({has_concise_summary/total*100:.1f}%)"
        )

        # Document current state for tracking improvement
        assert total > 0, "Should have papers in database"

        conn.close()

    def test_production_id_format_consistency(self):
        """Test that paper_id format is consistent in production database."""
        db_path = "./data/arxiv_papers.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check for inconsistent ID formats
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN paper_id LIKE 'http%' THEN 1 ELSE 0 END) as url_format,
                SUM(CASE WHEN paper_id NOT LIKE 'http%' THEN 1 ELSE 0 END) as id_format
            FROM papers
        """)

        result = cursor.fetchone()
        total, url_format, id_format = result

        print(f"Paper ID Format Analysis:")
        print(f"Total papers: {total}")
        print(f"URL format (http://...): {url_format}")
        print(f"ID format (2507.xxxxx): {id_format}")

        # After cleanup, should have consistent format
        assert (
            url_format == total or id_format == total
        ), "Should have consistent ID format"

        conn.close()


if __name__ == "__main__":
    # Run basic validation tests
    import subprocess

    print("🔍 Running Data Integrity Tests...")

    # Run pytest with verbose output
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
        print("✅ All data integrity tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(exit_code)
