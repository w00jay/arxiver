import logging
import sqlite3

logging.basicConfig(level=logging.INFO)


def create_connection(database):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        logging.error(e)
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
        interested BOOLEAN DEFAULT 0,
        authors TEXT,
        published DATE,
        categories TEXT,
        arxiv_url TEXT,
        pdf_url TEXT,
        abstract_embedding BLOB,
        citation_count INTEGER DEFAULT 0,
        related_papers TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_rating INTEGER,
        notes TEXT,
        tags TEXT,
        read_status TEXT DEFAULT 'unread',
        importance_score REAL DEFAULT 0.0
    );
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)

        # Create indexes for better performance
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers(categories)"
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers(authors)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_read_status ON papers(read_status)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_importance ON papers(importance_score)"
        )

        conn.commit()
    except sqlite3.Error as e:
        logging.error(e)


def add_interested_db_column(conn):
    sql = "ALTER TABLE papers ADD COLUMN interested BOOLEAN DEFAULT 0"
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logging.error(e)


def insert_article(conn, article):
    """Insert a new article into the papers table.

    Args:
        conn: Database connection
        article: Tuple or dict with paper information
    """
    if isinstance(article, dict):
        # Handle dict input with all fields
        sql = """INSERT OR IGNORE INTO papers(
            paper_id, title, summary, updated, authors, published, 
            categories, arxiv_url, pdf_url, interested
        ) VALUES(?,?,?,?,?,?,?,?,?,?)"""
        values = (
            article.get("paper_id"),
            article.get("title"),
            article.get("summary"),
            article.get("updated"),
            article.get("authors"),
            article.get("published"),
            article.get("categories"),
            article.get("arxiv_url"),
            article.get("pdf_url"),
            article.get("interested", 0),
        )
    else:
        # Handle tuple input (legacy)
        sql = "INSERT OR IGNORE INTO papers(paper_id,title,summary,updated) VALUES(?,?,?,?)"
        values = article

    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logging.error(e)
        return None


def update_concise_summary(conn, paper_id, concise_summary):
    """
    Update an article with its concise summary in the papers table.
    """
    sql = "UPDATE papers SET concise_summary = ? WHERE paper_id = ?"
    try:
        cur = conn.cursor()
        cur.execute(sql, (concise_summary, paper_id))
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logging.error(e)


def get_recent_entries(conn, limit=10):
    """Get recent papers with all metadata."""
    sql = """SELECT paper_id, title, summary, concise_summary, updated,
                    authors, published, categories, arxiv_url, pdf_url,
                    importance_score, read_status
             FROM papers ORDER BY updated DESC LIMIT ?"""
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)
        return []


def get_paper_by_id(conn, paper_id):
    """Get all information about a paper by its ID."""
    sql = """SELECT paper_id, title, summary, concise_summary, updated, interested,
                    authors, published, categories, arxiv_url, pdf_url,
                    citation_count, related_papers, last_updated, user_rating,
                    notes, tags, read_status, importance_score
             FROM papers WHERE paper_id = ?"""
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (paper_id,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        logging.error(e)
        return None


def get_paper_by_base_id(conn, base_paper_id):
    """Get the latest version of a paper by its base ID (without version).

    For example, if base_paper_id is "1706.03762", this will find
    "1706.03762v7" if that's the latest version in the database.
    """
    sql = """SELECT paper_id, title, summary, concise_summary, updated, interested,
                    authors, published, categories, arxiv_url, pdf_url,
                    citation_count, related_papers, last_updated, user_rating,
                    notes, tags, read_status, importance_score
             FROM papers 
             WHERE paper_id LIKE ? 
             ORDER BY paper_id DESC 
             LIMIT 1"""
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        # Match papers that start with the base ID
        cursor.execute(sql, (f"{base_paper_id}%",))
        return cursor.fetchone()
    except sqlite3.Error as e:
        logging.error(e)
        return None


def get_recent_papers_since_days(conn, days=2):
    """Get recent papers with all metadata since specified days back."""
    sql = """SELECT paper_id, title, summary, concise_summary, interested,
                    authors, published, categories, arxiv_url, pdf_url,
                    importance_score, read_status, tags, notes
             FROM papers 
             WHERE updated >= date('now', '-{} day')
             ORDER BY updated DESC""".format(int(days))
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)
        return []


def get_papers_between(conn, start_date, end_date):
    sql = "SELECT paper_id, title, summary, concise_summary FROM papers WHERE updated BETWEEN? AND?"
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (start_date, end_date))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)


def search_papers_by_author(conn, author_name, limit=50):
    """Search papers by author name (case-insensitive partial match)."""
    sql = """SELECT paper_id, title, summary, concise_summary, authors, published, 
                    categories, arxiv_url, importance_score
             FROM papers 
             WHERE authors LIKE ? 
             ORDER BY published DESC 
             LIMIT ?"""
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (f"%{author_name}%", limit))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)
        return []


def search_papers_by_category(conn, category, limit=50, days_back=None):
    """Search papers by category with optional date filtering."""
    if days_back:
        sql = """SELECT paper_id, title, summary, concise_summary, authors, published, 
                        categories, arxiv_url, importance_score
                 FROM papers 
                 WHERE categories LIKE ? 
                 AND published >= date('now', '-{} day')
                 ORDER BY published DESC 
                 LIMIT ?""".format(int(days_back))
    else:
        sql = """SELECT paper_id, title, summary, concise_summary, authors, published, 
                        categories, arxiv_url, importance_score
                 FROM papers 
                 WHERE categories LIKE ? 
                 ORDER BY published DESC 
                 LIMIT ?"""

    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (f"%{category}%", limit))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)
        return []


def update_paper_metadata(conn, paper_id, metadata):
    """Update paper metadata (authors, categories, urls, etc.)."""
    sql = """UPDATE papers SET 
             authors = ?, published = ?, categories = ?, 
             arxiv_url = ?, pdf_url = ?, last_updated = CURRENT_TIMESTAMP
             WHERE paper_id = ?"""
    try:
        cur = conn.cursor()
        cur.execute(
            sql,
            (
                metadata.get("authors"),
                metadata.get("published"),
                metadata.get("categories"),
                metadata.get("arxiv_url"),
                metadata.get("pdf_url"),
                paper_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logging.error(e)
        return False


def update_paper_importance(conn, paper_id, importance_score):
    """Update the importance score of a paper."""
    sql = "UPDATE papers SET importance_score = ? WHERE paper_id = ?"
    try:
        cur = conn.cursor()
        cur.execute(sql, (importance_score, paper_id))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logging.error(e)
        return False


def mark_paper_as_read(conn, paper_id):
    """Mark a paper as read."""
    sql = "UPDATE papers SET read_status = 'read' WHERE paper_id = ?"
    try:
        cur = conn.cursor()
        cur.execute(sql, (paper_id,))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logging.error(e)
        return False


def add_paper_notes(conn, paper_id, notes):
    """Add or update notes for a paper."""
    sql = "UPDATE papers SET notes = ? WHERE paper_id = ?"
    try:
        cur = conn.cursor()
        cur.execute(sql, (notes, paper_id))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logging.error(e)
        return False


def add_paper_tags(conn, paper_id, tags):
    """Add or update tags for a paper (comma-separated)."""
    sql = "UPDATE papers SET tags = ? WHERE paper_id = ?"
    try:
        cur = conn.cursor()
        cur.execute(sql, (tags, paper_id))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logging.error(e)
        return False


def search_papers_by_tags(conn, tags, limit=50):
    """Search papers by tags (comma-separated)."""
    sql = """SELECT paper_id, title, summary, concise_summary, authors, published, 
                    categories, arxiv_url, tags, importance_score
             FROM papers 
             WHERE tags LIKE ? 
             ORDER BY importance_score DESC, published DESC 
             LIMIT ?"""
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(sql, (f"%{tags}%", limit))
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(e)
        return []
