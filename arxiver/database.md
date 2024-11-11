# Database Module Documentation

This module provides SQLite database operations for managing arXiv paper data in the Arxiver project.

## Overview

The database module handles all database interactions for storing and retrieving arXiv paper information. It uses SQLite3 for data persistence and includes functionality for paper management with features like storing paper metadata, concise summaries, and interest tracking.

## Database Schema

The papers table has the following structure:

```sql
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    concise_summary TEXT,
    updated TEXT NOT NULL,
    interested BOOLEAN DEFAULT 0
);
```

## Functions

### Connection Management
- `create_connection(database)`: Creates a connection to the SQLite database
- `create_table(conn)`: Creates the papers table if it doesn't exist

### Data Operations
- `insert_article(conn, article)`: Inserts a new article into the database
- `update_concise_summary(conn, paper_id, concise_summary)`: Updates an article's concise summary
- `add_interested_db_column(conn)`: Adds the interested column to the papers table

### Query Operations
- `get_recent_entries(conn, limit=10)`: Retrieves the most recent papers
- `get_paper_by_id(conn, paper_id)`: Retrieves a specific paper by ID
- `get_recent_papers_since_days(conn, days=2)`: Gets papers from the last N days
- `get_papers_between(conn, start_date, end_date)`: Retrieves papers between two dates

## Usage Example

```python
from arxiver.database import create_connection, create_table, insert_article

# Create database connection
conn = create_connection("arxiv_papers.db")

# Initialize database table
create_table(conn)

# Insert a new article
article = ("paper123", "Example Title", "Paper summary", "2024-01-01")
insert_article(conn, article)
```

## Error Handling

All database operations include error handling with logging. Errors are logged using Python's built-in logging module, which is configured at the INFO level.

## Dependencies
- sqlite3: Built-in Python SQLite database interface
- logging: Built-in Python logging module
