import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .database import (
        add_interested_db_column,
        create_connection,
        create_table,
        get_paper_by_id,
        get_recent_entries,
        get_recent_papers_since_days,
        insert_article,
        update_concise_summary,
    )
except ImportError:
    from database import (
        add_interested_db_column,
        create_connection,
        create_table,
        get_paper_by_id,
        get_recent_entries,
        get_recent_papers_since_days,
        insert_article,
        update_concise_summary,
    )


@patch("sqlite3.connect")
def test_create_connection(mock_connect):
    create_connection("test.db")
    mock_connect.assert_called_once_with("test.db")


@patch("database.create_connection")
def test_create_table(mock_connection):
    conn = mock_connection.return_value
    create_table(conn)
    conn.cursor().execute.assert_called_once()


@patch("database.create_connection")
def test_add_interested_db_column(mock_connection):
    conn = mock_connection.return_value
    add_interested_db_column(conn)
    conn.cursor().execute.assert_called_once_with(
        "ALTER TABLE papers ADD COLUMN interested BOOLEAN DEFAULT 0"
    )
    conn.commit.assert_called_once()


@patch("database.create_connection")
def test_insert_article(mock_connection):
    conn = mock_connection.return_value
    insert_article(conn, ("id123", "Title", "Summary", "2020-01-01"))
    conn.cursor().execute.assert_called_once_with(
        "INSERT OR IGNORE INTO papers(paper_id,title,summary,updated) VALUES(?,?,?,?)",
        ("id123", "Title", "Summary", "2020-01-01"),
    )
    conn.commit.assert_called_once()


@patch("database.create_connection")
def test_update_concise_summary(mock_connection):
    conn = mock_connection.return_value
    update_concise_summary(conn, "id123", "Concise Summary")
    conn.cursor().execute.assert_called_once_with(
        "UPDATE papers SET concise_summary = ? WHERE paper_id = ?",
        ("Concise Summary", "id123"),
    )
    conn.commit.assert_called_once()


@patch("database.create_connection")
def test_get_recent_entries(mock_connection):
    conn = mock_connection.return_value
    get_recent_entries(conn, 5)
    conn.cursor().execute.assert_called_once_with(
        "SELECT paper_id, title, summary, concise_summary FROM papers ORDER BY updated DESC LIMIT ?",
        (5,),
    )


@patch("database.create_connection")
def test_get_paper_by_id(mock_connection):
    conn = mock_connection.return_value
    get_paper_by_id(conn, "id123")
    conn.cursor().execute.assert_called_once_with(
        "SELECT paper_id, title, summary, concise_summary FROM papers WHERE paper_id = ?",
        ("id123",),
    )


@patch("database.create_connection")
def test_get_recent_papers_since_days(mock_connection):
    conn = mock_connection.return_value
    get_recent_papers_since_days(conn, 3)
    conn.cursor().execute.assert_called_once_with(
        "SELECT paper_id, title, summary, concise_summary, interested FROM papers WHERE updated >= date('now', '-3 day')"
    )
