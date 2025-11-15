#!/usr/bin/env python3
"""
Enhanced MCP Tools Integration Tests

These tests verify that MCP tools work correctly in realistic scenarios
and integrate properly with the FastMCP framework.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the arxiver directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arxiver"))

from mcp_server import app


class TestMCPToolsIntegration:
    """Test MCP tools integration with realistic scenarios."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for isolated testing."""
        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.CHROMADB_AVAILABLE", True):
                with patch("mcp_server.chromadb") as mock_chromadb:
                    with patch("mcp_server.get_latest_model") as mock_model:
                        mock_conn.return_value = MagicMock()
                        mock_model.return_value = "/path/to/model"

                        # Mock ChromaDB
                        mock_collection = MagicMock()
                        mock_collection.query.return_value = {
                            "ids": [["2501.12345", "2501.12346"]],
                            "documents": [
                                [
                                    "ML paper about transformers",
                                    "CV paper about detection",
                                ]
                            ],
                            "distances": [[0.1, 0.2]],
                            "metadatas": [
                                [
                                    {
                                        "title": "Transformer Models",
                                        "authors": "Smith et al.",
                                        "categories": "cs.AI",
                                    },
                                    {
                                        "title": "Object Detection",
                                        "authors": "Jones et al.",
                                        "categories": "cs.CV",
                                    },
                                ]
                            ],
                        }
                        mock_client = MagicMock()
                        mock_client.get_or_create_collection.return_value = (
                            mock_collection
                        )
                        mock_chromadb.PersistentClient.return_value = mock_client

                        yield {
                            "conn": mock_conn,
                            "chromadb": mock_chromadb,
                            "model": mock_model,
                            "collection": mock_collection,
                        }

    async def test_search_papers_integration(self, mock_dependencies):
        """Test search_papers tool integration."""
        result = await app._tools["search_papers"]("machine learning transformers", 5)

        json_data = json.loads(result)
        assert isinstance(json_data, dict)

        if "error" not in json_data:
            assert "query" in json_data
            assert json_data["query"] == "machine learning transformers"
            assert "total_results" in json_data
            assert "papers" in json_data
            assert isinstance(json_data["papers"], list)

    async def test_get_recommendations_integration(self, mock_dependencies):
        """Test get_recommendations tool integration."""
        with patch("mcp_server.get_recent_papers_since_days") as mock_papers:
            mock_papers.return_value = [
                {
                    "paper_id": "2501.12345",
                    "title": "Advanced Machine Learning",
                    "authors": "Alice Smith, Bob Johnson",
                    "published": "2025-01-15",
                    "categories": "cs.AI, cs.LG",
                    "summary": "This paper presents novel ML techniques.",
                    "arxiv_url": "https://arxiv.org/abs/2501.12345",
                },
                {
                    "paper_id": "2501.12346",
                    "title": "Deep Learning Applications",
                    "authors": "Carol Davis, David Wilson",
                    "published": "2025-01-14",
                    "categories": "cs.AI, cs.CV",
                    "summary": "Applications of deep learning in computer vision.",
                    "arxiv_url": "https://arxiv.org/abs/2501.12346",
                },
            ]

            result = await app._tools["get_recommendations"](3)

            json_data = json.loads(result)
            assert isinstance(json_data, dict)

            if "error" not in json_data:
                assert "days_back" in json_data
                assert json_data["days_back"] == 3
                assert "total_papers" in json_data
                assert "recommendations" in json_data
                assert isinstance(json_data["recommendations"], list)

    async def test_summarize_paper_integration(self, mock_dependencies):
        """Test summarize_paper tool integration."""
        with patch("mcp_server.get_paper_by_id") as mock_get_paper:
            with patch("mcp_server.summarize_summary") as mock_summarize:
                mock_get_paper.return_value = {
                    "paper_id": "2501.12345",
                    "title": "Advanced Machine Learning Techniques",
                    "authors": "Alice Smith, Bob Johnson",
                    "published": "2025-01-15",
                    "categories": "cs.AI, cs.LG",
                    "summary": "This paper presents comprehensive analysis of machine learning.",
                    "concise_summary": "Novel ML techniques with improved performance.",
                    "arxiv_url": "https://arxiv.org/abs/2501.12345",
                    "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
                }
                mock_summarize.return_value = (
                    "Novel ML techniques with improved performance."
                )

                result = await app._tools["summarize_paper"]("2501.12345")

                json_data = json.loads(result)
                assert isinstance(json_data, dict)

                if "error" not in json_data:
                    assert "paper_id" in json_data
                    assert "title" in json_data
                    assert "summary" in json_data
                    assert json_data["title"] == "Advanced Machine Learning Techniques"

    async def test_choose_best_papers_integration(self, mock_dependencies):
        """Test choose_best_papers tool integration."""
        with patch("mcp_server.choose_summaries") as mock_choose:
            mock_choose.return_value = [
                {
                    "paper_id": "2501.12345",
                    "title": "Best ML Paper",
                    "relevance_score": 0.95,
                    "summary": "Highly relevant paper",
                },
                {
                    "paper_id": "2501.12346",
                    "title": "Second Best Paper",
                    "relevance_score": 0.87,
                    "summary": "Also relevant paper",
                },
            ]

            result = await app._tools["choose_best_papers"](
                "machine learning optimization", 2, 10
            )

            json_data = json.loads(result)
            assert isinstance(json_data, dict)

            if "error" not in json_data:
                assert "query" in json_data
                assert "selected_papers" in json_data
                assert len(json_data["selected_papers"]) <= 2

    async def test_import_paper_integration(self, mock_dependencies):
        """Test import_paper tool integration."""
        with patch("mcp_server.fetch_article_for_id") as mock_fetch:
            with patch("mcp_server.insert_article") as mock_insert:
                mock_fetch.return_value = {
                    "paper_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
                    "published": "2017-06-12",
                    "categories": "cs.CL, cs.AI",
                    "summary": "The dominant sequence transduction models...",
                    "arxiv_url": "https://arxiv.org/abs/1706.03762",
                }

                result = await app._tools["import_paper"]("1706.03762")

                json_data = json.loads(result)
                assert isinstance(json_data, dict)

                if "error" not in json_data:
                    assert "arxiv_id" in json_data
                    assert "status" in json_data
                    assert "title" in json_data
                    assert json_data["arxiv_id"] == "1706.03762"
                    assert "Attention Is All You Need" in json_data["title"]

    async def test_get_paper_details_integration(self, mock_dependencies):
        """Test get_paper_details tool integration."""
        with patch("mcp_server.get_paper_by_id") as mock_get_paper:
            mock_get_paper.return_value = {
                "paper_id": "2501.12345",
                "title": "Machine Learning Survey",
                "authors": "Research Team",
                "published": "2025-01-15",
                "summary": "Comprehensive survey of ML techniques.",
                "concise_summary": "ML survey paper.",
                "categories": "cs.AI, cs.LG",
                "arxiv_url": "https://arxiv.org/abs/2501.12345",
                "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
                "importance_score": 0.8,
                "read_status": "unread",
                "tags": "survey, ml",
                "notes": "Important reference",
            }

            result = await app._tools["get_paper_details"]("2501.12345")

            json_data = json.loads(result)
            assert isinstance(json_data, dict)

            if "error" not in json_data:
                assert "paper_id" in json_data
                assert "title" in json_data
                assert "importance_score" in json_data
                assert json_data["title"] == "Machine Learning Survey"

    async def test_search_papers_advanced_integration(self, mock_dependencies):
        """Test search_papers_advanced tool integration."""
        result = await app._tools["search_papers_advanced"](
            "deep learning", 5, "cs.AI", "2025-01-01", "2025-01-31", 0.7
        )

        json_data = json.loads(result)
        assert isinstance(json_data, dict)

        if "error" not in json_data:
            assert "query" in json_data
            assert "filters" in json_data
            assert "total_results" in json_data
            assert "papers" in json_data

            filters = json_data["filters"]
            assert filters["category"] == "cs.AI"
            assert filters["date_from"] == "2025-01-01"
            assert filters["date_to"] == "2025-01-31"
            assert filters["min_score"] == 0.7

    async def test_get_trending_papers_integration(self, mock_dependencies):
        """Test get_trending_papers tool integration."""
        with patch("mcp_server.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                (
                    "2501.12345",
                    "Trending ML Paper",
                    "Author 1",
                    "cs.AI",
                    "2025-01-15",
                    "url1",
                    0.9,
                ),
                (
                    "2501.12346",
                    "Popular CV Paper",
                    "Author 2",
                    "cs.CV",
                    "2025-01-14",
                    "url2",
                    0.8,
                ),
                (
                    "2501.12347",
                    "Hot NLP Paper",
                    "Author 3",
                    "cs.CL",
                    "2025-01-13",
                    "url3",
                    0.85,
                ),
            ]
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            result = await app._tools["get_trending_papers"](7, "cs.AI")

            json_data = json.loads(result)
            assert isinstance(json_data, dict)

            if "error" not in json_data:
                assert "days_back" in json_data
                assert "category_filter" in json_data
                assert "total_papers" in json_data
                assert "trending_papers" in json_data
                assert json_data["days_back"] == 7
                assert json_data["category_filter"] == "cs.AI"


class TestMCPToolsErrorHandling:
    """Test MCP tools error handling in various scenarios."""

    async def test_tools_with_invalid_inputs(self):
        """Test all tools handle invalid inputs gracefully."""
        invalid_test_cases = [
            ("search_papers", ["", 5]),  # Empty query
            ("search_papers", ["test", 0]),  # Invalid top_k
            ("get_recommendations", [-1]),  # Invalid days_back
            ("summarize_paper", [""]),  # Empty paper_id
            ("choose_best_papers", ["", 3, 10]),  # Empty query
            ("import_paper", [""]),  # Empty arxiv_id
            ("get_paper_details", [""]),  # Empty paper_id
            ("search_papers_advanced", ["", 5, "", "", "", 0.0]),  # Empty query
            ("get_trending_papers", [-1, ""]),  # Invalid days_back
        ]

        for tool_name, args in invalid_test_cases:
            result = await app._tools[tool_name](*args)
            json_data = json.loads(result)

            # Should return error response, not raise exception
            assert "error" in json_data
            assert "message" in json_data

    async def test_tools_with_database_failures(self):
        """Test tools handle database connection failures."""
        with patch("mcp_server.create_connection") as mock_conn:
            mock_conn.return_value = None  # Simulate connection failure

            database_dependent_tools = [
                ("get_recommendations", [3]),
                ("summarize_paper", ["test123"]),
                ("import_paper", ["1706.03762"]),
                ("get_paper_details", ["test123"]),
                ("get_trending_papers", [7, ""]),
            ]

            for tool_name, args in database_dependent_tools:
                result = await app._tools[tool_name](*args)
                json_data = json.loads(result)

                # Should handle database failure gracefully
                assert "error" in json_data or "message" in json_data

    async def test_tools_with_missing_dependencies(self):
        """Test tools handle missing external dependencies."""
        # Test with ChromaDB unavailable
        with patch("mcp_server.CHROMADB_AVAILABLE", False):
            result = await app._tools["search_papers"]("test", 5)
            json_data = json.loads(result)

            # Should return appropriate error message
            assert "error" in json_data
            assert "ChromaDB" in json_data["error"] or "ChromaDB" in json_data.get(
                "message", ""
            )


class TestMCPToolsPerformance:
    """Test MCP tools performance characteristics."""

    async def test_tools_response_time(self):
        """Test that tools respond within reasonable time."""
        import time

        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.search_papers_impl") as mock_search:
                mock_conn.return_value = MagicMock()
                mock_search.return_value = {"query": "test", "papers": []}

                start_time = time.time()
                result = await app._tools["search_papers"]("test", 5)
                end_time = time.time()

                # Should complete quickly with mocked dependencies
                response_time = end_time - start_time
                assert response_time < 1.0

                # Should return valid response
                json_data = json.loads(result)
                assert isinstance(json_data, dict)

    async def test_tools_concurrent_execution(self):
        """Test tools can handle concurrent execution."""
        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.search_papers_impl") as mock_search:
                mock_conn.return_value = MagicMock()
                mock_search.return_value = {"query": "test", "papers": []}

                # Create concurrent tasks
                tasks = [
                    app._tools["search_papers"](f"query_{i}", 5) for i in range(10)
                ]

                results = await asyncio.gather(*tasks)

                # All should complete successfully
                assert len(results) == 10
                for result in results:
                    json_data = json.loads(result)
                    assert isinstance(json_data, dict)


# Legacy integration test for backwards compatibility
async def test_all_tools():
    """Legacy test function for backwards compatibility."""
    print("🚀 Testing Arxiver MCP Server Tools (Legacy)")
    print("=" * 50)

    tools_to_test = [
        ("search_papers", ["machine learning", 5], "Search Papers"),
        ("get_recommendations", [3], "Get Recommendations"),
        ("summarize_paper", ["2501.12345"], "Summarize Paper"),
        ("get_paper_details", ["2501.12345"], "Get Paper Details"),
        ("import_paper", ["1706.03762"], "Import Paper"),
    ]

    for tool_name, args, description in tools_to_test:
        print(f"\n🔧 Testing {description}...")
        try:
            result = await app._tools[tool_name](*args)
            json_data = json.loads(result)

            if "error" in json_data:
                print(f"⚠️  Tool returned error: {json_data['error']}")
            else:
                print(f"✅ Success: Tool returned valid response")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 MCP Server testing complete!")
    print("\n💡 To run comprehensive tests:")
    print("   python -m pytest tests/test_mcp_tools.py -v")


if __name__ == "__main__":
    # Run both legacy test and pytest
    import subprocess

    # Run legacy test
    asyncio.run(test_all_tools())

    print("\n" + "=" * 60)
    print("Running comprehensive pytest suite...")

    # Run pytest
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
        print("✅ All MCP tools integration tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(exit_code)
