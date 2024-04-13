import sqlite3


def create_connection(database):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        print(e)
    return conn


def create_table(conn):
    """Create a table from the create_table_sql statement."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        summary TEXT,
        concise_summary TEXT,
        updated TEXT NOT NULL
    );
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def insert_article(conn, article):
    """Insert a new article into the papers table."""
    sql = """INSERT OR IGNORE INTO papers(paper_id,title,summary,updated) VALUES(?,?,?,?)"""
    cur = conn.cursor()
    cur.execute(sql, article)
    conn.commit()
    return cur.lastrowid


def update_concise_summary(conn, paper_id, concise_summary):
    """
    Update an article with its concise summary in the papers table.
    """
    sql = """ UPDATE papers SET concise_summary = ? WHERE paper_id = ? """
    cur = conn.cursor()
    cur.execute(sql, (concise_summary, paper_id))
    conn.commit()


def get_recent_entries(conn, limit=10):
    cursor = conn.cursor()
    query = "SELECT paper_id, title, summary, concise_summary FROM papers ORDER BY updated DESC LIMIT ?"
    cursor.execute(query, (limit,))
    return cursor.fetchall()
