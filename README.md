# arxiver

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> **Status: Pre-release** - A sophisticated arXiv paper management and discovery system with AI-powered recommendations

## 🎯 Overview

**arxiver** is an intelligent research tool designed to help ML researchers, AI practitioners, and academics stay up-to-date with the rapidly evolving arXiv landscape. It combines semantic search, machine learning recommendations, and AI-powered summarization to streamline academic paper discovery and management.

### 🚀 Key Features

- **📊 Semantic Search**: Vector-based similarity search using ChromaDB and sentence transformers
- **🤖 AI Recommendations**: TensorFlow-powered models predict papers of interest based on your reading history
- **📝 Intelligent Summarization**: LLM-generated concise summaries for quick paper evaluation
- **🔍 Smart Paper Selection**: AI-powered filtering to find the most relevant papers from large result sets
- **🛠 Model Context Protocol**: Enhanced MCP server with FastMCP best practices, middleware, and type safety
- **🔒 Security & Logging**: Comprehensive middleware for input validation, security, and request/response logging
- **⚡ Modern Stack**: FastAPI backend, ChromaDB vector store, UV package management, Pydantic models
- **🎪 Multiple Interfaces**: CLI tools, REST API, Streamlit UI, and production-ready MCP server

### 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   arXiv API     │    │   Streamlit UI  │    │  Claude/AI      │
│                 │    │                 │    │  Assistant      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Ingestion │  │   Search    │  │     MCP Server          │ │
│  │   Pipeline  │  │   Engine    │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐    ┌─────────────┐  ┌──────────────┐
    │ SQLite  │    │  ChromaDB   │  │ TensorFlow   │
    │Database │    │Vector Store │  │   Models     │
    └─────────┘    └─────────────┘  └──────────────┘
```

## 🔧 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Git**: For cloning the repository
- **Optional**: CUDA-compatible GPU for faster TensorFlow inference

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the fastest Python package installer and resolver.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/woojay/arxiver.git
cd arxiver

# Install dependencies and create virtual environment
uv sync

# Activate the environment (if needed)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/woojay/arxiver.git
cd arxiver

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Method 3: Using conda

```bash
# Create conda environment
conda create -n arxiver python=3.11
conda activate arxiver

# Clone and install
git clone https://github.com/woojay/arxiver.git
cd arxiver
pip install -e .
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required: OpenAI API key for summarization and LLM features
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom arXiv API settings
ARXIV_RESULTS_PER_PAGE=100
ARXIV_MAX_RESULTS=500

# Optional: Database settings
DATABASE_PATH=./data/arxiver.db
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Optional: Model settings
MODEL_PATH=./predictor/
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Database Initialization

```bash
# Initialize the database and vector store
uv run python -c "from arxiver.database import init_db; init_db()"
```

## 🚀 Quick Start

### 1. Start the FastAPI Server

```bash
# Using the CLI wrapper (from project root)
uv run python arxiver/main.py webserver

# Or using uvicorn directly
uv run uvicorn arxiver.main:app --reload --port 8000
```

### 2. Ingest Recent Papers

```bash
# Ingest papers from the last 7 days
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'

# Or use the CLI (from project root)
uv run python arxiver/main.py ingest --days 7
```

### 3. Start the Streamlit UI (Optional)

```bash
cd ui
uv run streamlit run arxiver_ui.py --server.port 8001
```

Visit http://localhost:8001 to access the web interface.

## 📖 Usage Examples

### REST API

```bash
# Search for papers on transformers
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanisms", "top_k": 10}'

# Get AI-powered recommendations
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"days_back": 3}'

# Summarize a specific paper
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "2404.04292"}'

# Get the best papers from a search
curl -X POST http://127.0.0.1:8000/choose \
  -H "Content-Type: application/json" \
  -d '{"query": "computer vision", "top_i": 5, "search_k": 50}'
```

### Python API

```python
import requests

# Search for papers
response = requests.post(
    "http://127.0.0.1:8000/query",
    json={"query": "large language models", "top_k": 5}
)
papers = response.json()

# Get recommendations
response = requests.post(
    "http://127.0.0.1:8000/recommend",
    json={"days_back": 7}
)
recommendations = response.json()
```

### CLI Interface

```bash
# Show available commands (from project root)
uv run python arxiver/main.py --help

# Ingest papers from specific date range
uv run python arxiver/main.py ingest --days 14

# Add interested column to database (for ML training)
uv run python arxiver/main.py add-interested-column
```

## 🤖 MCP Server Integration

arxiver includes a production-ready **Model Context Protocol (MCP)** server that enables AI assistants like Claude to interact with your paper database directly. The server implements FastMCP best practices with comprehensive middleware, type safety, and security features.

### ✨ FastMCP Enhancements

The MCP server has been enhanced with modern FastMCP features:

- **🛡️ Security Middleware**: Input validation, malicious pattern detection, and configurable security policies
- **📊 Logging Middleware**: Comprehensive request/response logging with sanitized parameters
- **🔒 Type Safety**: Full Pydantic model integration with structured responses and error handling
- **⚡ Performance**: Execution time tracking and optimized response formatting
- **📋 Standards Compliance**: Full MCP protocol compliance with enhanced error responses

For detailed information about the enhancements, see [FASTMCP_ENHANCEMENTS.md](./FASTMCP_ENHANCEMENTS.md).

📋 **Latest Updates**: See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.

### Starting the MCP Server

```bash
# Method 1: Direct Python execution (from project root)
uv run python arxiver/mcp_server.py

# Method 2: Using shell script (from project root)
./run_mcp_server.sh
```

### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_papers` | Semantic similarity search | `query`, `top_k` |
| `get_recommendations` | ML-powered recommendations | `days_back` |
| `summarize_paper` | Generate paper summaries | `paper_id` |
| `choose_best_papers` | AI-powered paper selection | `query`, `top_i`, `search_k` |
| `import_paper` | Import specific papers | `arxiv_id` |
| `get_paper_details` | Detailed paper information | `paper_id` |

### MCP Usage Examples

```bash
# Search for papers (using MCP CLI if available)
mcp call search_papers '{"query": "reinforcement learning", "top_k": 10}'

# Get recommendations for the past week
mcp call get_recommendations '{"days_back": 7}'

# Import a specific paper
mcp call import_paper '{"arxiv_id": "2404.04292"}'
```

### Claude Desktop Integration

For detailed MCP integration instructions with Claude Desktop, see [README-MCP.md](README-MCP.md).

## 🧪 Development Setup

### Running Tests

```bash
# Install development dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=arxiver --cov-report=html

# Run specific test file
uv run pytest tests/test_database.py -v
```

### Code Quality

```bash
# Format code
uv run black arxiver/ tests/

# Type checking
uv run mypy arxiver/

# Linting
uv run ruff check arxiver/
```

### Model Training

```bash
# Train interest prediction models
cd predictor
uv run python predict_interest.py

# The training will create timestamped model files
# Latest model is automatically used for recommendations
```

## 📊 API Reference

### FastAPI Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/ingest` | POST | Bulk ingest papers | `{"days": int}` |
| `/query` | POST | Semantic search | `{"query": str, "top_k": int}` |
| `/recommend` | POST | Get recommendations | `{"days_back": int}` |
| `/summarize` | POST | Summarize paper | `{"paper_id": str}` |
| `/choose` | POST | AI paper selection | `{"query": str, "top_i": int, "search_k": int}` |
| `/import` | POST | Import specific paper | `{"arxiv_id": str}` |

For complete API documentation, visit http://127.0.0.1:8000/docs when the server is running.

## 🗂 Project Structure

```
arxiver/
├── arxiver/                 # Main package
│   ├── __init__.py         # Package initialization
│   ├── main.py             # CLI and FastAPI server
│   ├── mcp_server.py       # MCP protocol server
│   ├── database.py         # SQLite database operations
│   ├── arxiv.py           # arXiv API integration
│   └── llm.py             # LLM and AI functionality
├── predictor/              # ML models and training
│   ├── predict_interest.py # Model training script
│   └── model-*.keras      # Trained TensorFlow models
├── ui/                     # Streamlit web interface
│   └── arxiver_ui.py      # Streamlit application
├── tests/                  # Test suite
│   ├── test_database.py   # Database tests
│   ├── test_llm.py        # LLM functionality tests
│   └── test_mcp_tools.py  # MCP server tests
├── data/                   # Data storage (created at runtime)
│   ├── arxiver.db         # SQLite database
│   └── chroma_db/         # ChromaDB vector store
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── README-MCP.md          # Detailed MCP documentation
└── run_mcp_server.sh      # MCP server startup script
```

## 🤝 Contributing

We welcome contributions! Please open an issue on GitHub to discuss major changes before submitting a pull request.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Format code: `uv run black arxiver/ tests/`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use arxiver in your research, please cite:

```bibtex
@software{arxiver,
  title={arxiver: Intelligent arXiv Paper Discovery and Management},
  author={Woojay Poynter},
  year={2025},
  url={https://github.com/woojay/arxiver}
}
```

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [ChromaDB](https://www.trychroma.com/), and [TensorFlow](https://tensorflow.org/)
- Uses [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- MCP integration powered by [Anthropic's MCP](https://modelcontextprotocol.io/)
- Package management with [uv](https://github.com/astral-sh/uv)

---

## 📋 Recent Updates

### Latest Changes (2025-11-01)
- **Performance Optimization**: Fixed recommendation endpoint with batch embedding retrieval (99.93% reduction in database queries)
- **Path Configuration**: Resolved relative path issues for production deployments
- **Import Fixes**: Corrected relative imports in vector_db module
- **Error Handling**: Fixed numpy array boolean ambiguity in embedding checks

### Previous Changes (2025-07-19)
- **Enhanced Database Schema**: Added comprehensive metadata fields (authors, categories, publication dates, etc.)
- **Fixed ChromaDB Issues**: Resolved vector database compatibility problems
- **Improved Search**: New author and category search capabilities
- **Better Error Handling**: Enhanced reliability and fallback mechanisms

For detailed change history, see [CHANGELOG.md](changelog/CHANGELOG.md).

### Migration & Issues Documentation

- **[Changelog](changelog/CHANGELOG.md)** - Complete version history and changes
- **[ChromaDB Issue Resolution (2025-07-19)](changelog/2025-07-19_chromadb_issue.md)** - Vector database compatibility fixes
- **[Database Migration (2025-07-19)](changelog/2025-07-19_database_migration_summary.md)** - Schema enhancement details  
- **[Vector DB Reconstruction (2025-07-19)](changelog/2025-07-19_vector_db_reconstruction.md)** - Database rebuild procedures
- **[Prevention Measures (2025-07-19)](changelog/2025-07-19_prevent_chromadb_issues.md)** - Safeguards to prevent future ChromaDB issues
- **[Critical Analysis (2025-07-19)](changelog/2025-07-19_critical_analysis.md)** - System failure analysis and fixes
- **[Comprehensive Review (2025-07-19)](changelog/2025-07-19_comprehensive_review_complete.md)** - Complete system review and testing documentation

## 🆘 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Clear uv cache if installation fails
uv cache clean

# Reinstall dependencies
rm -rf .venv
uv sync
```

**Database Issues:**
```bash
# Reset database
rm -f data/arxiver.db data/chroma_db/
uv run python -c "from arxiver.database import init_db; init_db()"
```

**ChromaDB Vector Database Issues:**
- If experiencing '_type' errors or embedding failures, see [ChromaDB Issue Resolution](changelog/2025-07-19_chromadb_issue.md)
- Complete vector database reconstruction may be required for corrupted databases
- Use `fill-missing-embeddings` endpoint to regenerate embeddings after fixes

**MCP Server Problems:**
- Ensure OpenAI API key is set in `.env`
- Check that the required ports are not in use (FastAPI: 8000, MCP server runs separately)
- Verify ChromaDB initialization
- Ensure database exists: `uv run python -c "from arxiver.database import init_db; init_db()"`

**Model Training Issues:**
- Ensure sufficient disk space for model files
- Check TensorFlow GPU installation if using GPU
- Verify training data exists in database

For more detailed troubleshooting, see [README-MCP.md](README-MCP.md) or open an issue on GitHub.

---

**Happy researching! 🚀📚**