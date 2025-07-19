#!/usr/bin/env python3
"""
MCP Performance Baseline Tests

These tests establish performance baselines for MCP server operations
and verify that the server meets acceptable performance criteria.
"""

import asyncio
import json
import os
import sys
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Add arxiver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arxiver"))

from mcp_server import app
# Note: MCPTestHelpers and MCPPerformanceHelpers will be implemented if needed


class TestMCPPerformanceBaselines:
    """Test MCP server performance baselines."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies for consistent performance testing."""
        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.CHROMADB_AVAILABLE", True):
                with patch("mcp_server.chromadb") as mock_chromadb:
                    with patch("mcp_server.get_latest_model") as mock_model:
                        mock_conn.return_value = MagicMock()
                        mock_model.return_value = "/path/to/model"
                        
                        # Mock ChromaDB for consistent performance
                        mock_collection = MagicMock()
                        mock_collection.query.return_value = {
                            "ids": [["paper1", "paper2", "paper3"]],
                            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
                            "distances": [[0.1, 0.2, 0.3]],
                            "metadatas": [[{}, {}, {}]]
                        }
                        mock_client = MagicMock()
                        mock_client.get_or_create_collection.return_value = mock_collection
                        mock_chromadb.PersistentClient.return_value = mock_client
                        
                        yield mock_conn

    async def test_search_papers_response_time(self, mock_dependencies):
        """Test that search_papers responds within acceptable time."""
        import time
        
        start_time = time.time()
        content, result = await app.call_tool("search_papers", {"query": "machine learning", "top_k": 5})
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 2 seconds with mocked dependencies
        assert response_time < 2.0, f"Response time too slow: {response_time:.3f}s"

    async def test_get_recommendations_response_time(self, mock_dependencies):
        """Test that get_recommendations responds within acceptable time."""
        with patch("mcp_server.get_recent_papers_since_days") as mock_papers:
            mock_papers.return_value = [
                {"paper_id": "test", "title": "Test", "summary": "Test"}
            ]
            
            import time
            start_time = time.time()
            content, result = await app.call_tool("get_recommendations", {"days_back": 3})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should respond within 3 seconds with ML model loading
            assert response_time < 3.0, f"Response time too slow: {response_time:.3f}s"

    async def test_summarize_paper_response_time(self, mock_dependencies):
        """Test that summarize_paper responds within acceptable time."""
        with patch("mcp_server.get_paper_by_id") as mock_get_paper:
            mock_get_paper.return_value = {
                "paper_id": "test123",
                "title": "Test Paper",
                "summary": "Test summary",
                "concise_summary": "Short summary"
            }
            
            import time
            start_time = time.time()
            content, result = await app.call_tool("summarize_paper", {"paper_id": "test123"})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should respond within 1 second for existing summary
            assert response_time < 1.0, f"Response time too slow: {response_time:.3f}s"

    async def test_get_paper_details_response_time(self, mock_dependencies):
        """Test that get_paper_details responds within acceptable time."""
        with patch("mcp_server.get_paper_by_id") as mock_get_paper:
            mock_get_paper.return_value = {
                "paper_id": "test123",
                "title": "Test Paper",
                "summary": "Test summary"
            }
            
            import time
            start_time = time.time()
            content, result = await app.call_tool("get_paper_details", {"paper_id": "test123"})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should respond within 1 second for database lookup
            assert response_time < 1.0, f"Response time too slow: {response_time:.3f}s"

    async def test_resource_response_time(self, mock_dependencies):
        """Test that resources respond within acceptable time."""
        with patch("mcp_server.get_recent_papers_since_days") as mock_papers:
            mock_papers.return_value = [
                {"paper_id": "test", "title": "Test", "authors": "Author"}
            ]
            
            import time
            start_time = time.time()
            result = await app.read_resource("arxiver://recent-papers")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Resources should respond within 1 second
            assert response_time < 1.0, f"Resource response time too slow: {response_time:.3f}s"
            assert isinstance(result, list)


class TestMCPConcurrentPerformance:
    """Test MCP server performance under concurrent load."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies for concurrent testing."""
        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.search_papers_impl") as mock_search:
                mock_conn.return_value = MagicMock()
                mock_search.return_value = {"query": "test", "papers": []}
                yield mock_conn

    async def test_concurrent_search_requests(self, mock_dependencies):
        """Test concurrent search requests performance."""
        # Prepare test arguments for concurrent requests
        args_list = [
            (f"query_{i}", 5) for i in range(10)
        ]
        
        # Execute concurrent requests
        import asyncio
        tasks = [
            app.call_tool("search_papers", {"query": query, "top_k": top_k})
            for query, top_k in args_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least 80% should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"
        assert len(successful) >= 8

    async def test_concurrent_mixed_requests(self, mock_dependencies):
        """Test concurrent mixed tool requests."""
        with patch("mcp_server.get_paper_by_id") as mock_get_paper:
            mock_get_paper.return_value = {
                "paper_id": "test", "title": "Test", "summary": "Test"
            }
            
            # Mix different tool calls
            async def mixed_calls():
                tasks = [
                    app.call_tool("search_papers", {"query": "query1", "top_k": 5}),
                    app.call_tool("get_paper_details", {"paper_id": "test123"}),
                    app.call_tool("search_papers", {"query": "query2", "top_k": 5}),
                    app.call_tool("get_paper_details", {"paper_id": "test456"}),
                    app.call_tool("search_papers", {"query": "query3", "top_k": 5}),
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)
            
            start_time = time.time()
            results = await mixed_calls()
            end_time = time.time()
            
            # Should complete all requests quickly
            total_time = end_time - start_time
            assert total_time < 5.0  # All 5 requests in under 5 seconds
            
            # Most should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) >= 4

    async def test_server_resource_usage(self, mock_dependencies):
        """Test server resource usage under load."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        tasks = [
            app.call_tool("search_papers", {"query": f"query_{i}", "top_k": 5})
            for i in range(20)
        ]
        
        await asyncio.gather(*tasks)
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.1f} MB"

    async def test_response_time_under_load(self, mock_dependencies):
        """Test that response times remain acceptable under load."""
        # Create background load
        background_tasks = [
            app.call_tool("search_papers", {"query": f"background_{i}", "top_k": 5})
            for i in range(5)
        ]
        
        # Start background tasks
        background_futures = [asyncio.create_task(task) for task in background_tasks]
        
        # Measure response time of foreground request
        start_time = time.time()
        content, result = await app.call_tool("search_papers", {"query": "foreground_query", "top_k": 5})
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Clean up background tasks
        for future in background_futures:
            if not future.done():
                future.cancel()
        
        # Response time should still be acceptable under load
        assert response_time < 3.0, f"Response time under load too high: {response_time:.3f}s"
        
        # Result should be valid
        json_data = json.loads(result['result'])
        assert isinstance(json_data, dict)


class TestMCPScalabilityLimits:
    """Test MCP server scalability limits."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for scalability testing."""
        with patch("mcp_server.create_connection") as mock_conn:
            with patch("mcp_server.search_papers_impl") as mock_search:
                mock_conn.return_value = MagicMock()
                mock_search.return_value = {"query": "test", "papers": []}
                yield mock_conn

    async def test_large_query_handling(self, mock_dependencies):
        """Test handling of large query inputs."""
        # Test with progressively larger queries
        query_sizes = [100, 1000, 5000]  # Characters
        
        for size in query_sizes:
            large_query = "machine learning " * (size // 16)  # Approximate size
            
            start_time = time.time()
            content, result = await app.call_tool("search_papers", {"query": large_query, "top_k": 5})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should handle large queries reasonably
            assert response_time < 5.0, f"Large query ({size} chars) took too long: {response_time:.3f}s"
            
            # Should return valid response
            json_data = json.loads(result['result'])
            assert isinstance(json_data, dict)

    async def test_high_top_k_values(self, mock_dependencies):
        """Test handling of high top_k values."""
        with patch("mcp_server.CHROMADB_AVAILABLE", True):
            with patch("mcp_server.chromadb") as mock_chromadb:
                # Mock large result set
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    "ids": [[f"paper{i}" for i in range(50)]],
                    "documents": [[f"Document {i}" for i in range(50)]],
                    "distances": [[i * 0.01 for i in range(50)]],
                    "metadatas": [[{} for i in range(50)]]
                }
                mock_client = MagicMock()
                mock_client.get_or_create_collection.return_value = mock_collection
                mock_chromadb.PersistentClient.return_value = mock_client
                
                # Test with high top_k
                content, result = await app.call_tool("search_papers", {"query": "test", "top_k": 50})
                json_data = json.loads(result['result'])
                
                if "error" not in json_data:
                    assert "papers" in json_data
                    # Should handle large result sets
                    assert len(json_data["papers"]) <= 50

    async def test_rapid_successive_requests(self, mock_dependencies):
        """Test handling of rapid successive requests."""
        # Make many requests in quick succession
        tasks = []
        for i in range(50):
            task = app.call_tool("search_papers", {"query": f"rapid_{i}", "top_k": 5})
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Should handle 50 requests reasonably quickly
        assert total_time < 10.0, f"50 rapid requests took too long: {total_time:.3f}s"
        
        # Most should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"


class TestMCPMemoryEfficiency:
    """Test MCP server memory efficiency."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for memory testing."""
        with patch("mcp_server.create_connection") as mock_conn:
            mock_conn.return_value = MagicMock()
            yield mock_conn

    async def test_memory_usage_with_large_responses(self, mock_dependencies):
        """Test memory usage with large response data."""
        # Create mock large dataset
        large_papers = [
            {
                "paper_id": f"paper{i}",
                "title": f"Paper {i} with very long title " * 10,
                "summary": "Long summary " * 100,
                "authors": f"Author {i}, Co-Author {i}",
                "categories": "cs.AI, cs.LG, cs.CV"
            }
            for i in range(100)
        ]
        
        with patch("mcp_server.search_papers_impl") as mock_search:
            mock_search.return_value = {
                "query": "test",
                "total_results": 100,
                "papers": large_papers
            }
            
            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            # Execute tool with large response
            content, result = await app.call_tool("search_papers", {"query": "test", "top_k": 100})
            
            # Measure memory after
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
            
            # Should not use excessive memory
            assert memory_used < 50 * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.1f} MB"
            
            # Response should be valid
            json_data = json.loads(result['result'])
            assert isinstance(json_data, dict)

    async def test_memory_cleanup_after_operations(self, mock_dependencies):
        """Test that memory is properly cleaned up after operations."""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(20):
            with patch("mcp_server.search_papers_impl") as mock_search:
                mock_search.return_value = {"query": f"test{i}", "papers": []}
                await app.call_tool("search_papers", {"query": f"test{i}", "top_k": 5})
        
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal
        assert memory_growth < 20 * 1024 * 1024, f"Memory leak detected: {memory_growth / 1024 / 1024:.1f} MB growth"


if __name__ == "__main__":
    # Run MCP performance baseline tests
    import subprocess

    print("⚡ Running MCP Performance Baseline Tests...")

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
        print("✅ All MCP performance baseline tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(exit_code)