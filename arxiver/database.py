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
        updated TEXT NOT NULL,
        interested BOOLEAN DEFAULT 0
    );
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def add_interested_db_column(conn):
    sql = """ALTER TABLE papers ADD COLUMN interested BOOLEAN DEFAULT 0"""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        print(e)


def insert_article(conn, article):
    """Insert a new article into the papers table."""
    sql = """INSERT OR IGNORE INTO papers(paper_id,title,summary,updated) VALUES(?,?,?,?)"""
    try:
        cur = conn.cursor()
        cur.execute(sql, article)
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        print(e)


def update_concise_summary(conn, paper_id, concise_summary):
    """
    Update an article with its concise summary in the papers table.
    """
    sql = """ UPDATE papers SET concise_summary = ? WHERE paper_id = ? """
    try:
        cur = conn.cursor()
        cur.execute(sql, (concise_summary, paper_id))
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        print(e)


def get_recent_entries(conn, limit=10):
    sql = "SELECT paper_id, title, summary, concise_summary FROM papers ORDER BY updated DESC LIMIT ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(e)

def get_paper_by_id(conn, paper_id):
    sql = "SELECT paper_id, title, summary, concise_summary FROM papers WHERE paper_id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (paper_id,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        print(e)
