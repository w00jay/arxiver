# Changelog

All notable changes to the Arxiver project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-07-19 - FastMCP Enhancement Release

### 🚀 Major Features

#### FastMCP Best Practices Implementation
- **Comprehensive Middleware Framework**: Implemented extensible middleware architecture with base `MCPMiddleware` class
- **Type Safety**: Full Pydantic model integration for all requests and responses
- **Security First**: Input validation, malicious pattern detection, and configurable security policies
- **Real-time Monitoring**: Request/response logging with sanitized parameters and performance metrics
- **Interactive Capabilities**: Progress reporting and user input elicitation during tool execution

#### New Middleware Components
- **LoggingMiddleware**: Comprehensive request/response logging with argument sanitization
- **SecurityMiddleware**: Input validation and malicious pattern detection (XSS, SQL injection)
- **ProgressMiddleware**: Real-time progress tracking for long-running operations
- **UserInteractionMiddleware**: Interactive user input elicitation with timeout support

### 🛠️ Enhanced Tools

#### Enhanced `import_paper`
- **Progress Tracking**: Step-by-step progress reporting (0% → 100%)
- **Operation IDs**: Unique operation identifiers for status tracking
- **Detailed Messaging**: Progress messages for each import step
- **Embedding Integration**: Automatic embedding generation when ChromaDB available

#### New Tools
- **`get_operation_status`**: Query status of long-running operations by operation ID
- **`interactive_paper_selection`**: Interactive paper selection with user choice prompts

### 🔒 Security & Monitoring

#### Security Features
- **Input Validation**: Configurable maximum input length (default: 10,000 characters)
- **Pattern Detection**: Blocks malicious patterns including:
  - XSS attempts (`<script>`, `javascript:`)
  - SQL injection patterns (`drop table`, `delete from`)
  - Other configurable security patterns
- **Security Logging**: Detailed logging of security events and blocked requests

#### Monitoring & Logging
- **Request Tracking**: Sequential request numbering and sanitized parameter logging
- **Performance Metrics**: Execution time tracking for all operations
- **Result Analysis**: Safe result information extraction for logging
- **Error Classification**: Structured error responses with types and context

### 📋 Type Safety & Error Handling

#### Pydantic Models
- **Paper**: Structured paper representation with validation
- **SearchResponse**: Standardized search operation responses with metadata
- **RecommendationResponse**: ML-based recommendation responses
- **SummaryResponse**: Paper summarization responses
- **PaperDetailsResponse**: Detailed paper information responses
- **ImportResponse**: Paper import operation responses with progress tracking
- **ErrorResponse**: Standardized error handling with timestamps and context

#### Enhanced Error Handling
- **Error Classification**: Categorized error types (validation_error, security_error, etc.)
- **Detailed Context**: Error responses include relevant operation details
- **Execution Metrics**: All responses include execution time measurements
- **Graceful Degradation**: Fallback behaviors for various failure scenarios

### 🏗️ Architecture Improvements

#### Middleware Pipeline
```
Request → Security Check → Logging → Progress/Interaction → Tool Execution → Logging → Response
```

#### Configuration
- **Environment Variables**: New configuration options for security and limits
- **Middleware Selection**: Configurable middleware instances
- **Global References**: Middleware instances available for tool integration

### 📊 Performance

#### Benchmarks
- **Type Safety**: ~1-2ms overhead per request
- **Logging Middleware**: ~0.5ms overhead per request
- **Security Middleware**: ~1ms overhead for pattern matching
- **Total Overhead**: ~2-4ms per request (negligible for typical use cases)

#### Optimizations
- **Caching**: Maintained existing caching mechanisms
- **Validation**: Efficient input validation with early termination
- **Logging**: Optimized sanitization and result analysis

### 📖 Documentation

#### New Documentation
- **FASTMCP_ENHANCEMENTS.md**: Comprehensive FastMCP implementation guide
- **Architecture Documentation**: Middleware pipeline and type safety flow diagrams
- **Configuration Guide**: Environment variables and middleware configuration
- **Usage Examples**: Progress reporting and interactive tool examples

#### Updated Documentation
- **README.md**: Updated with FastMCP enhancement highlights
- **README-MCP.md**: Enhanced tool documentation with new features
- **Code Comments**: Comprehensive inline documentation for all new components

### 🧪 Testing

#### Test Coverage
- **Protocol Compliance**: All 15 MCP protocol tests passing
- **Security Testing**: Malicious input detection and blocking
- **Progress Tracking**: Operation status and progress reporting
- **Type Safety**: Pydantic model validation
- **Integration**: End-to-end workflow testing

#### Test Results
- ✅ MCP protocol compliance: 15/15 tests passing
- ✅ Security middleware: Successfully blocks malicious input
- ✅ Progress reporting: Operation tracking and status queries working
- ✅ Type safety: Pydantic validation functioning correctly
- ✅ Performance: Minimal overhead confirmed

### 🔧 Configuration

#### New Environment Variables
```bash
ARXIVER_MAX_INPUT_LENGTH=10000  # Maximum input length for security validation
```

#### Middleware Configuration
```python
middleware_instances = [
    LoggingMiddleware(),           # Request/response logging  
    progress_middleware,          # Progress tracking and reporting
    user_interaction_middleware,  # User input elicitation
    SecurityMiddleware()          # Input validation and security
]
```

### 🚧 Breaking Changes
- **None**: All changes are backward compatible
- **Response Format**: Enhanced with additional metadata (execution_time_ms, operation_id)
- **Error Responses**: Now include error_type and timestamp fields

### 🐛 Bug Fixes
- **Pydantic Compatibility**: Updated to use Pydantic V2 `field_validator`
- **Performance Tests**: Fixed to use FastMCP API instead of internal attributes
- **Import Process**: Enhanced error handling and progress reporting

### ⚡ Performance
- **Maintained**: Existing performance characteristics preserved
- **Enhanced**: Added execution time tracking without performance impact
- **Optimized**: Efficient middleware pipeline with minimal overhead

---

## [1.0.0] - 2025-01-06 - Initial Release

### 🚀 Added

#### Core MCP Server
- **Model Context Protocol**: Full MCP server implementation
- **Tool Interface**: 8 comprehensive tools for arXiv research
- **FastAPI Integration**: RESTful API alongside MCP server
- **Database Management**: SQLite-based paper storage and management

#### Research Tools
- **`search_papers`**: Semantic search using ChromaDB and sentence transformers
- **`get_recommendations`**: ML-powered personalized paper recommendations
- **`summarize_paper`**: AI-generated concise summaries using OpenAI
- **`choose_best_papers`**: LLM-powered paper ranking and selection
- **`import_paper`**: On-demand paper import from arXiv
- **`get_paper_details`**: Detailed paper information retrieval
- **`search_papers_advanced`**: Advanced search with filtering options
- **`get_trending_papers`**: Trending paper discovery

#### AI & ML Integration
- **TensorFlow Models**: Trained recommendation models for interest prediction
- **Vector Search**: ChromaDB integration for semantic similarity
- **OpenAI Integration**: GPT-powered summarization and paper selection
- **Embedding Pipeline**: Automated paper embedding generation

#### Data Management
- **arXiv API**: Comprehensive arXiv paper fetching and processing
- **SQLite Database**: Efficient local paper storage with metadata
- **Vector Database**: ChromaDB for semantic search capabilities
- **Model Storage**: TensorFlow model management and versioning

#### User Interfaces
- **MCP Server**: Primary interface for AI assistants (Claude, etc.)
- **Streamlit UI**: Web interface for interactive research
- **CLI Tools**: Command-line interface for batch operations
- **REST API**: FastAPI server for programmatic access

#### Configuration & Setup
- **Environment Variables**: Flexible configuration system
- **Docker Support**: Containerized deployment options
- **UV Package Management**: Modern Python dependency management
- **Pre-commit Hooks**: Code quality and formatting enforcement

### 📋 Documentation
- **README.md**: Comprehensive project documentation
- **README-MCP.md**: Detailed MCP server guide
- **API Documentation**: Tool schemas and usage examples
- **Setup Guides**: Installation and configuration instructions

### 🧪 Testing
- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflow testing
- **MCP Protocol Tests**: Protocol compliance verification
- **Performance Tests**: Baseline performance measurement

### 🔧 Infrastructure
- **GitHub Actions**: Automated testing and deployment
- **Pre-commit**: Code quality enforcement
- **UV**: Modern Python package management
- **Type Hints**: Comprehensive type annotations

---

## Versioning Strategy

- **Major Version (X.0.0)**: Breaking changes, major feature additions
- **Minor Version (X.Y.0)**: New features, non-breaking changes
- **Patch Version (X.Y.Z)**: Bug fixes, minor improvements

## Contributing

See [Contributing Guidelines](CONTRIBUTING.md) for information on how to contribute to this project.

## Support

- 📧 Email: tenaciouswp@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/arxiver/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/arxiver/discussions)