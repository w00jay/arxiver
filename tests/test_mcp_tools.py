#!/usr/bin/env python3
"""
Test script for Arxiver MCP Server tools
"""

import asyncio
import os
import sys

# Add the arxiver directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "arxiver"))

from mcp_server import (
    get_paper_details,
    get_recommendations,
    import_paper,
    ingest_recent_papers,
    summarize_paper,
)


async def test_all_tools():
    """Test all MCP tools with real data."""
    print("🚀 Testing Arxiver MCP Server Tools")
    print("=" * 50)

    # Test 1: Get paper details
    print("\n1️⃣ Testing get_paper_details...")
    try:
        result = await get_paper_details(
            {"paper_id": "http://arxiv.org/abs/2507.02864v1"}
        )
        print(f"✅ Success: {result['title'][:60]}...")
        print(f"   Status: {result['status']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Summarize paper
    print("\n2️⃣ Testing summarize_paper...")
    try:
        result = await summarize_paper(
            {"paper_id": "http://arxiv.org/abs/2507.02864v1"}
        )
        print(f"✅ Success: {result['status']}")
        if result.get("concise_summary"):
            print(f"   Summary: {result['concise_summary'][:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Get recommendations
    print("\n3️⃣ Testing get_recommendations...")
    try:
        result = await get_recommendations({"days_back": 3})
        print(f"✅ Success: {result}")
        rec_count = result.get(
            "recommendations_count", len(result.get("recommendations", []))
        )
        print(f"   Found {rec_count} recommendations")
        if result.get("total_papers_analyzed"):
            print(f"   Analyzed {result['total_papers_analyzed']} papers")
        if result.get("recommendations"):
            print(f"   First rec: {result['recommendations'][0]['title'][:50]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 4: Import a specific paper
    print("\n4️⃣ Testing import_paper...")
    try:
        result = await import_paper({"arxiv_id": "1706.03762"})
        print(f"✅ Success: {result['status']}")
        print(f"   Paper: {result['title'][:50]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 MCP Server testing complete!")
    print("\n💡 Next steps:")
    print("   1. Install dependencies: uv sync")
    print("   2. Run: uv run python arxiver/mcp_server.py")
    print("   3. Configure in Claude Desktop MCP settings")


if __name__ == "__main__":
    asyncio.run(test_all_tools())
