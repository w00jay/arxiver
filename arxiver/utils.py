"""Utility functions for the arxiver package."""

import re


def clean_paper_id(paper_id: str) -> str:
    """Clean and normalize arXiv paper ID.

    Handles various formats:
    - http://arxiv.org/abs/1706.03762v7 -> 1706.03762v7
    - https://arxiv.org/abs/1706.03762 -> 1706.03762
    - 1706.03762v7 -> 1706.03762v7
    - 1706.03762 -> 1706.03762

    Args:
        paper_id: Raw paper ID in any format

    Returns:
        Normalized paper ID (e.g., "1706.03762" or "1706.03762v7")
    """
    if not paper_id:
        return ""

    # Remove URL parts if present
    if "arxiv.org" in paper_id:
        paper_id = paper_id.split("/")[-1]

    # Remove any query parameters
    if "?" in paper_id:
        paper_id = paper_id.split("?")[0]

    # Clean whitespace
    paper_id = paper_id.strip()

    return paper_id


def get_paper_id_without_version(paper_id: str) -> str:
    """Get paper ID without version number.

    Examples:
    - 1706.03762v7 -> 1706.03762
    - 1706.03762 -> 1706.03762

    Args:
        paper_id: Paper ID possibly with version

    Returns:
        Paper ID without version
    """
    clean_id = clean_paper_id(paper_id)
    # Remove version suffix if present
    return re.sub(r"v\d+$", "", clean_id)


def get_latest_version_query(paper_id: str) -> str:
    """Generate SQL pattern to find latest version of a paper.

    Args:
        paper_id: Base paper ID without version

    Returns:
        SQL LIKE pattern to match all versions
    """
    base_id = get_paper_id_without_version(paper_id)
    return f"{base_id}%"
