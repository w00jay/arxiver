# Arxiver Project Guidelines

## Project Overview
Arxiver is an arXiv paper recommendation and management system with:
- FastAPI server for paper ingestion, search, and recommendations
- SQLite database for paper metadata
- ChromaDB vector database for semantic search
- MCP (Model Context Protocol) server integration
- ML-based paper recommendation system

## Development Workflow

### Before Making Changes
1. **Run existing tests** to understand current state:
   ```bash
   # Core module tests (priority)
   uv run pytest arxiver/test_arxiv.py arxiver/test_database.py arxiver/test_llm.py -v

   # Full test suite (171 tests, takes ~10-15 minutes)
   uv run pytest -v
   ```

2. **Check server health**:
   ```bash
   # Start server
   cd arxiver
   uvicorn main:app --reload --port 8000

   # Test endpoints
   curl http://127.0.0.1:8000/recommend
   curl http://127.0.0.1:8000/fill-missing-embeddings
   curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"query_text": "machine learning"}'
   ```

### Making Changes

#### Code Standards
- **No emojis** unless explicitly requested
- **Minimal comments** - code should be self-documenting
- **Type hints** required for all functions
- **Edit existing files** rather than creating new ones
- **No documentation files** unless explicitly requested

#### Import Structure
All modules use try/except pattern for compatibility:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .module import function  # Relative import first
except ImportError:
    from module import function   # Fall back to absolute
```

#### Database Paths
- **SQLite**: `/home/woojay/P/ML/arxiver/data/arxiv_papers.db`
- **ChromaDB**: `/home/woojay/P/ML/arxiver/data/arxiv_embeddings.chroma`
- **Never use relative paths** - always compute absolute paths from `__file__`

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PAPERS_DB = os.path.join(PROJECT_ROOT, "data", "arxiv_papers.db")
```

### Testing Requirements

#### Critical Tests (Must Pass)
Located in `arxiver/` directory:
- `test_arxiv.py` - arXiv API fetching
- `test_database.py` - Database operations
- `test_llm.py` - LLM summarization

#### Test Mocking
Always mock `time.sleep` to avoid delays:
```python
from unittest.mock import patch

@patch('time.sleep', return_value=None)
def test_function(mock_sleep):
    # test code
```

#### Before Committing
1. Run core tests: `uv run pytest arxiver/test_*.py -v`
2. Ensure server starts without errors
3. Test at least one endpoint manually
4. Check no unintended files are staged

### After Implementation

#### Testing Checklist
- [ ] Core module tests pass (`arxiver/test_*.py`)
- [ ] Server starts successfully
- [ ] At least one endpoint tested manually
- [ ] No new import errors
- [ ] Database paths correct

#### Documentation
- Update `README.md` with significant changes
- Log changes in appropriate format (commit message is sufficient)
- **Do not create** separate documentation files unless requested

### Staging Changes

```bash
# Review changes
git status
git diff

# Stage specific files
git add arxiver/arxiv.py
git add arxiver/main.py
git add arxiver/chromadb_manager.py
# ... etc

# Create commit
git commit -m "Fix: Adjust arXiv rate limiting and import structure

- Updated retry backoff to comply with arXiv 3-second requirement
- Fixed relative import issues across modules
- Resolved ChromaDB version compatibility (1.0.15 -> 0.5.20)
- Fixed database path resolution for ChromaDB manager
- Updated test imports and mocked time.sleep delays"
```

## Known Issues

### MCP Tests (93+ failures)
Status: **NOT CRITICAL - Deferred**
- Most failures due to import structure changes
- MCP functionality works in production
- Will be addressed in future session

### Test Performance
- Full test suite takes 10-15 minutes
- Use subset testing during development
- Mock `time.sleep` in new tests to avoid delays

## Project Structure

```
arxiver/
├── arxiver/           # Main application code
│   ├── arxiv.py      # arXiv API interactions
│   ├── database.py   # SQLite operations
│   ├── main.py       # FastAPI server
│   ├── mcp_server.py # MCP integration
│   ├── chromadb_manager.py  # Vector DB manager
│   ├── vector_db.py  # Vector DB operations
│   ├── llm.py        # LLM interactions
│   └── test_*.py     # Core module tests
├── tests/            # Integration tests
├── data/             # Databases (gitignored)
│   ├── arxiv_papers.db
│   └── arxiv_embeddings.chroma/
├── predictor/        # ML model files
└── backup/           # Database backups
```

## Common Commands

```bash
# Start server
cd arxiver
uvicorn main:app --reload --port 8000

# Run core tests
uv run pytest arxiver/test_*.py -v

# Run specific test
uv run pytest arxiver/test_arxiv.py::test_fetch_article_for_id -v

# Code quality
uv run ruff check .
uv run ruff format .
uv run mypy .

# Database operations
sqlite3 data/arxiv_papers.db "SELECT COUNT(*) FROM papers;"
```

## Emergency Recovery

### ChromaDB Corruption
```bash
# Backup old database
mv data/arxiv_embeddings.chroma backup/arxiv_embeddings.chroma.$(date +%Y%m%d_%H%M%S)

# Regenerate (will create new database automatically)
curl -X GET http://127.0.0.1:8000/fill-missing-embeddings
```

### SQLite Issues
```bash
# Check integrity
sqlite3 data/arxiv_papers.db "PRAGMA integrity_check;"

# Restore from backup
cp backup/arxiv_papers_backup_*.db data/arxiv_papers.db
```

## Version Requirements
- Python: 3.12+
- ChromaDB: 0.5.20 (NOT 1.0.15)
- FastAPI, TensorFlow, sentence-transformers (see pyproject.toml)

## Contact & Support
- GitHub Issues: https://github.com/yourusername/arxiver/issues
- Documentation: README.md
