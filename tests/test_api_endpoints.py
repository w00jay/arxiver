#!/usr/bin/env python3
"""
API Endpoint Tests for Arxiver FastAPI Application

These tests validate all API endpoints for correct functionality,
error handling, and data integrity.
"""

import json
import os
import sqlite3
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add arxiver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arxiver"))

from database import (
    create_connection,
    create_table,
    insert_article,
    update_concise_summary,
)
from main import app


class TestAPIEndpoints:
    """Test FastAPI endpoints functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        conn = create_connection(db_path)
        create_table(conn)

        # Add some test data
        test_papers = [
            {
                "paper_id": "http://arxiv.org/abs/2507.12345v1",
                "title": "Test ML Paper",
                "summary": "This is a comprehensive test of machine learning methods.",
                "updated": "2025-07-17T10:30:00Z",
                "authors": "John Doe, Jane Smith",
                "published": "2025-07-17",
                "categories": "cs.AI, cs.LG",
                "arxiv_url": "http://arxiv.org/abs/2507.12345v1",
                "pdf_url": "http://arxiv.org/pdf/2507.12345v1.pdf",
                "interested": 0,
            },
            {
                "paper_id": "http://arxiv.org/abs/2507.12346v1",
                "title": "Another Test Paper",
                "summary": "This paper explores different aspects of AI research.",
                "updated": "2025-07-16T15:20:00Z",
                "authors": "Alice Johnson, Bob Wilson",
                "published": "2025-07-16",
                "categories": "cs.AI, cs.CV",
                "arxiv_url": "http://arxiv.org/abs/2507.12346v1",
                "pdf_url": "http://arxiv.org/pdf/2507.12346v1.pdf",
                "interested": 0,
            },
        ]

        for paper in test_papers:
            insert_article(conn, paper)

        # Add concise summary for first paper
        update_concise_summary(
            conn, "http://arxiv.org/abs/2507.12345v1", "Concise ML summary"
        )

        conn.close()

        yield db_path

        if os.path.exists(db_path):
            os.unlink(db_path)

    @patch("main.PAPERS_DB")
    def test_ingest_endpoint_success(self, mock_db_path, client, temp_db):
        """Test /ingest endpoint successful operation."""
        mock_db_path.__str__ = lambda x: temp_db

        with patch("main.ingest_process") as mock_ingest:
            response = client.post(
                "/ingest", json={"start_date": "2025-07-17", "days": 2}
            )

        assert response.status_code == 200
        data = response.json()
        assert "Ingestion process started" in data["message"]
        assert "2 days" in data["message"]

        # Verify ingest_process was called with correct parameters
        mock_ingest.assert_called_once_with("2025-07-17", 2)

    def test_ingest_endpoint_default_params(self, client):
        """Test /ingest endpoint with default parameters."""
        with patch("main.ingest_process") as mock_ingest:
            response = client.post("/ingest", json={})

        assert response.status_code == 200
        # Should use default days value
        mock_ingest.assert_called_once()

    def test_ingest_endpoint_validation(self, client):
        """Test /ingest endpoint input validation."""
        # Test with invalid JSON
        response = client.post("/ingest", data="invalid json")
        assert response.status_code == 422  # Validation error

    def test_summarize_endpoint_success(self, client, temp_db):
        """Test /summarize endpoint successful operation."""
        with patch("main.PAPERS_DB", temp_db):
            with patch("main.summarize_summary") as mock_summarize:
                mock_summarize.return_value = "Generated concise summary"

                response = client.post(
                    "/summarize", json={"paper_id": "http://arxiv.org/abs/2507.12346v1"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Generated a new concise summary."
        assert data["concise_summary"] == "Generated concise summary"

    def test_summarize_endpoint_existing_summary(self, client, temp_db):
        """Test /summarize endpoint when summary already exists."""
        with patch("main.PAPERS_DB", temp_db):
            response = client.post(
                "/summarize", json={"paper_id": "http://arxiv.org/abs/2507.12345v1"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Concise summary already exists."
        assert data["concise_summary"] == "Concise ML summary"

    def test_summarize_endpoint_paper_not_found(self, client, temp_db):
        """Test /summarize endpoint with non-existent paper."""
        with patch("main.PAPERS_DB", temp_db):
            response = client.post(
                "/summarize", json={"paper_id": "http://arxiv.org/abs/9999.99999v1"}
            )

            assert response.status_code == 404
            assert "Paper not found" in response.json()["detail"]

    @patch("main.chromadb_manager")
    def test_query_endpoint_success(self, mock_chromadb_manager, client):
        """Test /query endpoint successful operation."""
        # Mock ChromaDB response
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [["paper1", "paper2"]],
            "documents": [["Doc 1", "Doc 2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"source": "arxiv"}, {"source": "arxiv"}]],
        }

        mock_chromadb_manager.get_collection_context.return_value.__enter__ = (
            lambda x: mock_collection
        )
        mock_chromadb_manager.get_collection_context.return_value.__exit__ = (
            lambda x, *args: None
        )

        response = client.post(
            "/query", json={"query_text": "machine learning", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "paper1"
        assert data[0]["summary"] == "Doc 1"
        assert data[0]["distance"] == 0.1

    @patch("main.chromadb_manager")
    def test_query_endpoint_empty_collection(self, mock_chromadb_manager, client):
        """Test /query endpoint with empty ChromaDB collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chromadb_manager.get_collection_context.return_value.__enter__ = (
            lambda x: mock_collection
        )
        mock_chromadb_manager.get_collection_context.return_value.__exit__ = (
            lambda x, *args: None
        )

        response = client.post("/query", json={"query_text": "machine learning"})

        assert response.status_code == 200
        data = response.json()
        assert (
            data["message"]
            == "No embeddings available. Please run /fill-missing-embeddings first."
        )
        assert data["results"] == []

    def test_query_endpoint_validation(self, client):
        """Test /query endpoint input validation."""
        # Missing required query_text
        response = client.post("/query", json={})
        assert response.status_code == 422

        # Empty query_text
        response = client.post("/query", json={"query_text": ""})
        assert response.status_code == 422

    @patch("main.PAPERS_DB")
    @patch("main.get_latest_model")
    @patch("main.tf.keras.models.load_model")
    def test_recommend_endpoint_success(
        self, mock_load_model, mock_get_latest, mock_db_path, client, temp_db
    ):
        """Test /recommend endpoint successful operation."""
        mock_db_path.__str__ = lambda x: temp_db
        mock_get_latest.return_value = "/path/to/model.keras"

        # Mock TensorFlow model
        mock_model = MagicMock()
        mock_model.predict.return_value = [[True], [False]]  # First paper recommended
        mock_load_model.return_value = mock_model

        # Mock embedding function
        with patch("main.get_embedding") as mock_get_embedding:
            mock_get_embedding.side_effect = [
                [0.1, 0.2, 0.3],  # First paper embedding
                [0.4, 0.5, 0.6],  # Second paper embedding
            ]

            response = client.get("/recommend?days_back=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Only first paper recommended
        assert "Test ML Paper" in data[0]["title"]

    @patch("main.get_latest_model")
    def test_recommend_endpoint_no_model(self, mock_get_latest, client):
        """Test /recommend endpoint when no model is found."""
        mock_get_latest.return_value = None

        response = client.get("/recommend")
        assert response.status_code == 404
        assert "No model file found" in response.json()["detail"]

    @patch("main.PAPERS_DB")
    def test_fill_missing_summaries_endpoint(self, mock_db_path, client, temp_db):
        """Test /fill-missing-summaries endpoint."""
        mock_db_path.__str__ = lambda x: temp_db

        with patch("main.summarize_article") as mock_summarize:
            response = client.get("/fill-missing-summaries")

        assert response.status_code == 200
        data = response.json()
        assert "missing_summaries" in data

        # Should call summarize_article for paper without summary
        mock_summarize.assert_called()

    @patch("main.chromadb_manager")
    def test_fill_missing_embeddings_endpoint(self, mock_chromadb_manager, client):
        """Test /fill-missing-embeddings endpoint."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 50
        mock_collection.get.return_value = {"ids": []}  # No existing embeddings
        mock_collection.add.return_value = None

        mock_chromadb_manager.health_check.return_value = True
        mock_chromadb_manager.get_collection_context.return_value.__enter__ = (
            lambda x: mock_collection
        )
        mock_chromadb_manager.get_collection_context.return_value.__exit__ = (
            lambda x, *args: None
        )

        with patch("main.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("paper1", "Summary 1"),
                ("paper2", "Summary 2"),
            ]
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            response = client.get("/fill-missing-embeddings")

        assert response.status_code == 200
        data = response.json()
        assert "embeddings_added" in data
        assert "total_papers_checked" in data

    @patch("main.PAPERS_DB")
    def test_import_endpoint_success(self, mock_db_path, client, temp_db):
        """Test /import endpoint successful operation."""
        mock_db_path.__str__ = lambda x: temp_db

        with patch("main.import_process") as mock_import:
            response = client.post("/import", json={"arxiv_id": "1706.03762"})

        assert response.status_code == 200
        data = response.json()
        assert "Import process started" in data["message"]
        assert "1706.03762" in data["message"]

        mock_import.assert_called_once_with("1706.03762")

    def test_import_endpoint_validation(self, client):
        """Test /import endpoint input validation."""
        # Missing required arxiv_id
        response = client.post("/import", json={})
        assert response.status_code == 422

        # Empty arxiv_id
        response = client.post("/import", json={"arxiv_id": ""})
        assert response.status_code == 422


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON in POST requests."""
        response = client.post(
            "/ingest",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test handling of missing Content-Type header."""
        response = client.post("/ingest", data='{"test": "data"}')
        # Should still work with proper JSON
        assert response.status_code in [200, 422]  # Depends on validation

    @patch("main.create_connection")
    def test_database_connection_failure(self, mock_create_connection, client):
        """Test handling of database connection failures."""
        mock_create_connection.return_value = None

        response = client.post("/summarize", json={"paper_id": "test_paper"})

        assert response.status_code == 500
        assert "Database connection error" in response.json()["detail"]

    @patch("main.chromadb_manager")
    def test_chromadb_connection_failure(self, mock_chromadb_manager, client):
        """Test handling of ChromaDB connection failures."""
        mock_chromadb_manager.get_collection_context.side_effect = Exception(
            "ChromaDB connection failed"
        )

        response = client.post("/query", json={"query_text": "test query"})

        assert response.status_code == 500
        assert "Query error" in response.json()["detail"]


class TestAPIPerformance:
    """Test API performance and resource usage."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_concurrent_requests_handling(self, client):
        """Test that API can handle concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            with patch("main.ingest_process"):
                response = client.post("/ingest", json={"days": 1})
                results.append(response.status_code)

        # Start multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)

    def test_large_query_handling(self, client):
        """Test handling of large query requests."""
        # Test with very long query text
        long_query = "machine learning " * 1000

        with patch("main.chromadb_manager") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 100
            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "distances": [[]],
                "metadatas": [[]],
            }
            mock_chromadb.get_collection_context.return_value.__enter__ = (
                lambda x: mock_collection
            )
            mock_chromadb.get_collection_context.return_value.__exit__ = (
                lambda x, *args: None
            )

            response = client.post(
                "/query", json={"query_text": long_query, "top_k": 10}
            )

        # Should handle gracefully (not crash)
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    # Run API endpoint tests
    import subprocess

    print("🌐 Running API Endpoint Tests...")

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
        print("✅ All API endpoint tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(exit_code)
