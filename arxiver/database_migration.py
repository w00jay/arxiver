#!/usr/bin/env python3
"""
Database migration script to add missing columns to the papers table.
This script safely adds new columns while preserving existing data.
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiver.database import create_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    return column_name in columns


def add_column_if_not_exists(conn, table_name, column_name, column_definition):
    """Add a column to a table if it doesn't already exist."""
    if not check_column_exists(conn, table_name, column_name):
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )
            conn.commit()
            logger.info(f"Added column '{column_name}' to table '{table_name}'")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding column '{column_name}': {e}")
            return False
    else:
        logger.info(f"Column '{column_name}' already exists in table '{table_name}'")
        return True


def migrate_database(db_path):
    """Run all database migrations."""
    conn = create_connection(db_path)
    if not conn:
        logger.error("Failed to connect to database")
        return False

    logger.info(f"Starting database migration for: {db_path}")

    # Define new columns to add
    new_columns = [
        ("authors", "TEXT"),
        ("published", "DATE"),
        ("categories", "TEXT"),
        ("arxiv_url", "TEXT"),
        ("pdf_url", "TEXT"),
        ("abstract_embedding", "BLOB"),
        ("citation_count", "INTEGER DEFAULT 0"),
        ("related_papers", "TEXT"),
        ("last_updated", "TIMESTAMP"),
        ("user_rating", "INTEGER"),
        ("notes", "TEXT"),
        ("tags", "TEXT"),
        ("read_status", "TEXT DEFAULT 'unread'"),
        ("importance_score", "REAL DEFAULT 0.0"),
    ]

    success = True
    for column_name, column_def in new_columns:
        if not add_column_if_not_exists(conn, "papers", column_name, column_def):
            success = False

    # Create indexes for better query performance
    try:
        cursor = conn.cursor()

        # Index on published date for temporal queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published)"
        )

        # Index on categories for category-based searches
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers(categories)"
        )

        # Index on authors for author searches
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers(authors)"
        )

        # Index on read_status for filtering
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_read_status ON papers(read_status)"
        )

        # Index on importance_score for sorting
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_importance ON papers(importance_score)"
        )

        conn.commit()
        logger.info("Created database indexes successfully")
    except sqlite3.Error as e:
        logger.error(f"Error creating indexes: {e}")
        success = False

    conn.close()
    return success


def display_schema(db_path):
    """Display the current schema of the papers table."""
    conn = create_connection(db_path)
    if not conn:
        return

    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = cursor.fetchall()

    logger.info("\nCurrent papers table schema:")
    logger.info("-" * 60)
    for col in columns:
        logger.info(
            f"  {col[1]:20} {col[2]:15} {'NOT NULL' if col[3] else 'NULL':8} {f'DEFAULT {col[4]}' if col[4] else ''}"
        )
    logger.info("-" * 60)

    conn.close()


if __name__ == "__main__":
    import os

    # Determine database path
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default path
        script_dir = Path(__file__).parent
        db_path = script_dir.parent / "data" / "arxiv_papers.db"

    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        sys.exit(1)

    # Run migration
    if migrate_database(db_path):
        logger.info("Migration completed successfully!")
        display_schema(db_path)
    else:
        logger.error("Migration failed!")
        sys.exit(1)
