# Arxiver MCP Server

[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-blue)](https://modelcontextprotocol.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

The **Arxiver MCP Server** transforms your arXiv research workflow by providing AI assistants with intelligent access to academic paper search, ML-based recommendations, and automated summarization. Built on the Model Context Protocol (MCP), it enables natural language interactions with your research database.

### Key Features

- 🔍 **Semantic Search**: Vector-based similarity search through thousands of papers (requires ChromaDB)
- 🤖 **ML Recommendations**: Personalized paper suggestions using trained models
- 📝 **AI Summarization**: Automated concise summaries of complex papers
- 🎯 **Smart Selection**: LLM-powered paper ranking and selection
- 🔄 **Real-time Import**: Import specific papers on-demand

**Note**: Some features like semantic search require optional dependencies (ChromaDB). The MCP server will function with graceful fallbacks if optional components are unavailable.

## Quick Start

### 1. Installation

```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip with editable install
pip install -e .
```

### 2. Setup Environment

```bash
# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

### 3. Initialize Database

```bash
# Initialize the database and create tables
uv run python -c "
from arxiver.database import create_connection, create_table
import os
os.makedirs('./data', exist_ok=True)
conn = create_connection('./data/arxiver.db')
create_table(conn)
conn.close()
print('Database initialized successfully')
"

# Optionally ingest some recent papers (from project root)
uv run python arxiver/main.py ingest --days 3
```

### 4. Test MCP Server

```bash
# Test the MCP server functionality
uv run python tests/test_mcp_tools.py

# Should show:
# ✅ Paper details successful
# ✅ Summarization successful  
# ✅ Import paper successful
# ⚠️ Recommendations (needs embeddings)
```

### 5. Start MCP Server

```bash
# Run the MCP server (from project root)
uv run python arxiver/mcp_server.py
```

## Integration with AI Assistants

### Claude Desktop Integration

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "arxiver": {
      "command": "uv",
      "args": ["run", "python", "arxiver/mcp_server.py"],
      "cwd": "/path/to/arxiver"
    }
  }
}
```

### Claude Code Integration

```bash
# Add to your Claude Code settings
claude-code config add-mcp-server arxiver uv run python arxiver/mcp_server.py
```

## Available Tools

### 🔍 `search_papers`
Search papers using semantic similarity based on embeddings.

**Requirements:** ChromaDB must be installed and configured. If ChromaDB is not available, this tool will return an error with guidance on installation.

**Parameters:**
- `query` (required): Search query (e.g., "machine learning transformers")
- `top_k` (optional): Number of results (default: 5, max: 50)

**Example:**
```
Find papers about "graph neural networks for drug discovery"
```

### 🤖 `get_recommendations` 
Get personalized paper recommendations using ML models trained on your research interests.

**How it works:**
1. **Data Collection**: Retrieves recent papers from the specified time period
2. **ML Scoring**: Uses trained TensorFlow models to predict your interest in each paper
3. **Embedding Integration**: Leverages paper embeddings for similarity-based scoring
4. **Intelligent Fallback**: Returns recent papers with default scores if ML models unavailable
5. **Ranked Results**: Papers sorted by prediction confidence with detailed metadata

**Parameters:**
- `days_back` (optional): Days to look back (default: 3, max: 30)

**Response Format:**
```json
{
  "days_back": 3,
  "total_papers": 347,
  "recommendations": [
    {
      "paper_id": "http://arxiv.org/abs/2507.02593v1",
      "title": "Paper Title",
      "authors": "",
      "published": "",
      "summary": "Detailed paper summary...",
      "prediction_score": 0.85
    }
  ]
}
```

**Note:** `authors` and `published` fields are currently empty in the database schema but may be populated in future versions.

**Example:**
```
Get my personalized recommendations for the last week
```

**Advanced Usage:**
```
Show me the most interesting papers from the past 5 days with scores above 0.7
```

### 📝 `summarize_paper`
Generate or retrieve concise summaries of specific papers.

**Parameters:**
- `paper_id` (required): arXiv ID or URL

**Example:**
```
Summarize the paper "Attention Is All You Need" (arXiv:1706.03762)
```

### 🎯 `choose_best_papers`
Use LLM to intelligently select the most relevant papers.

**Parameters:**
- `query` (required): Search query
- `top_i` (optional): Number of papers to select (default: 3)
- `search_k` (optional): Papers to search through (default: 20)

**Example:**
```
Choose the 3 best papers about "vision transformers" from the top 50 results
```

### 📥 `import_paper`
Import specific papers from arXiv into your database.

**Parameters:**
- `arxiv_id` (required): arXiv ID (e.g., "2404.04292")

**Example:**
```
Import the paper 1706.03762 into my database
```


### 📊 `get_paper_details`
Get detailed information about papers in your database.

**Parameters:**
- `paper_id` (required): arXiv ID or URL

**Example:**
```
Get details for paper http://arxiv.org/abs/2404.04292v1
```

## Usage Examples

### Research Workflow Examples

#### Literature Review
```
Assistant: I'm starting a literature review on transformer architectures. Can you help me find relevant papers?

You: Search for papers about "transformer architecture attention mechanisms"

[Returns semantic search results with similarity scores]

Then: Choose the 3 best papers from those results for my specific research focus

[AI assistant automatically selects the most relevant papers using LLM analysis]
```

#### Daily Research Updates
```
Assistant: What are the latest interesting papers in my field?

You: Get my personalized recommendations for the last 3 days

[Returns ML-based recommendations tailored to your research interests]

Then: Summarize the paper about "Vision Transformers for Medical Imaging"

[Provides concise, AI-generated summary of the paper]
```

#### Targeted Research
```
Assistant: I heard about an interesting paper on arXiv ID 2404.04292. Can you analyze it for me?

You: Import paper 2404.04292 and summarize it

[Imports the paper and generates a concise summary]

Then: Get detailed information about this paper

[Provides comprehensive paper details including links and metadata]
```

### Advanced Usage Patterns

#### Bulk Literature Discovery
```python
# Through AI assistant conversation:
"Ingest recent papers from the last week, then get recommendations and choose the top 5 most relevant to my research on 'multimodal learning'"
```

#### Research Trend Analysis
```python
# Combine multiple tools for comprehensive analysis:
"Search for papers on 'large language models', choose the best 10, then summarize each one"
```

## Architecture

### System Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   AI Assistant      │    │   MCP Server        │    │   Data Layer        │
│   (Claude/GPT)      │<-->│   (arxiver)         │<-->│   (SQLite/Chroma)   │
│                     │    │                     │    │                     │
│ - Natural Language  │    │ - Tool Handlers     │    │ - Papers Database   │
│ - Tool Calling      │    │ - Search Logic      │    │ - Vector Embeddings │
│ - Response Format   │    │ - ML Models         │    │ - ML Models         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### ML Recommendation System

The `get_recommendations` tool uses a sophisticated ML pipeline:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Recent Papers     │    │   ML Pipeline       │    │   Ranked Results    │
│   (SQLite Query)    │───▶│   (TensorFlow)      │───▶│   (Scored Papers)   │
│                     │    │                     │    │                     │
│ - Paper Metadata    │    │ - Embedding Lookup  │    │ - Prediction Scores │
│ - Time Filtering    │    │ - Model Inference   │    │ - Sorted by Interest│
│ - Batch Processing  │    │ - Score Calculation │    │ - Metadata Enhanced │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

**Model Architecture:**
- **Input**: 384-dimensional paper embeddings (MiniLM-L6-v2)
- **Architecture**: Dense neural network with dropout layers
- **Output**: Single probability score (0-1) indicating research interest
- **Training**: Supervised learning on user-labeled paper preferences
- **Fallback**: Default scoring (0.8) when models unavailable

**Model Selection:**
- Automatically selects the latest trained model from `/predictor/model-*.keras`
- Handles legacy model compatibility with custom loading functions
- Graceful degradation when no models are available

### Data Flow

1. **User Query** → AI Assistant processes natural language
2. **Tool Selection** → MCP Server receives structured tool calls
3. **Data Processing** → Server queries databases and runs ML models
4. **Results** → Structured responses sent back to assistant
5. **Natural Response** → AI formats results for user

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Database settings
DATABASE_PATH=./data/arxiver.db        # SQLite database
CHROMA_PERSIST_DIRECTORY=./data/chroma_db  # Vector database

# Optional - arXiv API settings
ARXIV_RESULTS_PER_PAGE=100
ARXIV_MAX_RESULTS=500

# Optional - Model settings
MODEL_PATH=./predictor/
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional - MCP server settings
LOOK_BACK_DAYS=3              # Default days for searches
```

### Directory Structure

```
arxiver/
├── arxiver/
│   ├── mcp_server.py         # Main MCP server
│   ├── main.py               # FastAPI arxiver server
│   ├── database.py           # Database operations
│   ├── llm.py                # LLM integration
│   ├── arxiv.py              # arXiv API wrapper
│   └── vector_db.py          # Vector operations
├── data/
│   ├── arxiver.db            # SQLite database
│   └── chroma_db/            # ChromaDB vectors
├── predictor/
│   └── model-*.keras         # Trained ML models
├── ui/
│   └── arxiver_ui.py         # Streamlit interface
└── README-MCP.md             # This documentation
```

## Database Schema

### Papers Table (SQLite)
```sql
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    concise_summary TEXT,
    updated DATETIME,
    interested INTEGER DEFAULT 0
);
```

### Embeddings Collection (ChromaDB)
```python
# Collection: "arxiver"
# Embedding Model: "all-MiniLM-L6-v2"
# Documents: Concise summaries
# Metadata: {"source": "arxiv"}
# IDs: arXiv paper URLs
```

## Troubleshooting

### Common Issues

#### MCP Server Won't Start
```bash
# Check Python path and dependencies
uv run python --version  # Should be 3.11+
uv pip list | grep mcp  # Should show mcp>=1.1.0

# Check database connectivity
uv run python -c "from arxiver.database import create_connection; print('DB OK' if create_connection('./data/arxiver.db') else 'DB Error')"
```

#### No Search Results
```bash
# Check if papers are ingested
uv run python -c "
from arxiver.database import create_connection
conn = create_connection('./data/arxiver.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM papers')
print(f'Papers in database: {cursor.fetchone()[0]}')
"

# Check if embeddings exist
uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db')
collection = client.get_collection('arxiver')
print(f'Embeddings count: {collection.count()}')
"
```

#### ML Recommendations Failing
```bash
# Check if ML models exist
ls -la ../predictor/model-*.keras

# Test model loading
uv run python -c "
import tensorflow as tf
from arxiver.mcp_server import get_latest_model
model_path = get_latest_model('../predictor')
print(f'Latest model: {model_path}')
model = tf.keras.models.load_model(model_path, compile=False)
print(f'Model loaded successfully: {model.input_shape}')
"

# Test the complete recommendation pipeline
uv run python -c "
import asyncio
from arxiver.mcp_server import get_recommendations_impl
result = asyncio.run(get_recommendations_impl(days_back=1))
print(f'Recommendations test: {len(result.get(\"recommendations\", []))} papers')
if 'error' in result:
    print(f'Error: {result[\"error\"]}')
"

# Check embeddings availability for scoring
uv run python -c "
try:
    import chromadb
    from arxiver.mcp_server import get_embedding
    # Test embedding retrieval
    embedding = get_embedding('test_paper_id')
    print(f'Embedding system available: {embedding is not None}')
except Exception as e:
    print(f'Embedding system error: {e}')
"
```

**Common ML Issues:**
- **No models found**: Run training pipeline or use fallback mode
- **Model loading errors**: Check TensorFlow version compatibility
- **Embedding lookup fails**: Verify ChromaDB setup and paper ingestion
- **Low prediction scores**: Models may need retraining with more data

#### LLM Summarization Issues
```bash
# Check OpenAI API key
uv run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
print(f'API key configured: {bool(key and len(key) > 10)}')
"

# Test LLM connection
uv run python -c "
from arxiver.llm import summarize_summary
result = summarize_summary('This is a test summary for checking LLM connectivity.')
print(f'LLM test successful: {bool(result)}')
"
```

### Performance Optimization

#### Database Indexes
```sql
-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_papers_updated ON papers(updated);
CREATE INDEX IF NOT EXISTS idx_papers_interested ON papers(interested);
```

#### Memory Management
```python
# For large databases, consider pagination
top_k = min(requested_k, 100)  # Limit search results
```

#### Embedding Cache
```python
# ChromaDB automatically caches embeddings
# Consider increasing cache size for large collections
```

## Development

### Running Tests
```bash
# Run existing tests
uv run python -m pytest arxiver/test_*.py

# Test MCP server specifically
uv run python -c "
import asyncio
from arxiver.mcp_server import search_papers
result = asyncio.run(search_papers({'query': 'test', 'top_k': 1}))
print(f'MCP test result: {result}')
"
```

### Adding New Tools

1. **Define Tool Schema** in `handle_list_tools()`
2. **Add Tool Handler** in `handle_call_tool()`
3. **Implement Function** following async patterns
4. **Update Documentation** in this README

Example:
```python
# In handle_list_tools()
Tool(
    name="my_new_tool",
    description="Description of what this tool does",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"},
        },
        "required": ["param1"],
    },
)

# In handle_call_tool()
elif name == "my_new_tool":
    result = await my_new_tool(arguments)

# New function
async def my_new_tool(arguments: dict) -> dict:
    param1 = arguments["param1"]
    # Implementation here
    return {"result": "success"}
```

## Comparison with Other Approaches

### vs. Direct arXiv API
- ✅ **Semantic Search**: Vector similarity vs. keyword matching
- ✅ **Personalization**: ML-based recommendations
- ✅ **Summarization**: AI-generated concise summaries
- ✅ **Persistence**: Local database with history

### vs. Manual Research
- ✅ **Automation**: Automated discovery and ingestion
- ✅ **Scale**: Process thousands of papers efficiently
- ✅ **Intelligence**: AI-powered relevance ranking
- ✅ **Integration**: Works within AI assistant workflow

### vs. Traditional Literature Review Tools
- ✅ **Natural Language**: Conversational interaction
- ✅ **Real-time**: Live data from arXiv
- ✅ **Extensible**: Easy to add new capabilities
- ✅ **Open Source**: Full control and customization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📧 **Email**: tenaciouswp@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/arxiver/issues)
- 📖 **Docs**: This README and code comments
- 💬 **Discussions**: Use GitHub Discussions for questions

## Changelog

### v1.0.0 (2025-01-06)
- ✅ Initial MCP server implementation
- ✅ Core recommendation and summarization tools  
- ✅ Paper import and database management
- ✅ AI summarization integration
- ✅ Comprehensive documentation and testing
- ⚠️ Vector search (requires ChromaDB installation)

---

**Happy Researching! 🚀📚**