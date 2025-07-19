# Changelog

All notable changes to the Arxiver project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-07-19

### Added
- **Enhanced Database Schema**: Added 14 new metadata columns including authors, published date, categories, arxiv_url, pdf_url, citation_count, related_papers, last_updated, user_rating, notes, tags, read_status, and importance_score
- **Author and Category Search**: New database functions `search_papers_by_author()` and `search_papers_by_category()` for enhanced paper discovery
- **Paper ID Utilities**: New `utils.py` module with paper ID normalization and cleaning functions
- **Enhanced MCP Server**: Updated MCP server to handle new metadata fields and provide richer paper information
- **Comprehensive Documentation**: Added migration guides, issue tracking, and reconstruction procedures

### Changed
- **Database Migration**: Safely migrated existing 228K+ papers to new schema while preserving all data
- **Enhanced Paper Parsing**: Updated `arxiv.py` with `parse_arxiv_entry()` function to extract complete metadata from arXiv API
- **Improved Error Handling**: Better fallback mechanisms and batch processing throughout the system
- **Vector Database Configuration**: Updated ChromaDB configuration for better consistency and reliability

### Fixed
- **ChromaDB Compatibility Issue**: Resolved '_type' errors and schema mismatches that prevented embeddings generation
- **Vector Database Corruption**: Complete reconstruction of ChromaDB to resolve data corruption issues
- **Metadata Configuration**: Fixed inconsistent ChromaDB collection metadata causing compatibility problems
- **Batch Processing**: Enhanced fill-missing-embeddings with proper batch processing and error handling

### Technical Details
- **Vector Database Reconstruction Required**: Due to data corruption, complete ChromaDB rebuild was necessary (see [Vector DB Reconstruction](2025-07-19_vector_db_reconstruction.md))
- **Database Schema Migration**: Backward-compatible migration preserving all existing data (see [Database Migration Summary](2025-07-19_database_migration_summary.md))
- **ChromaDB Issue Resolution**: Comprehensive fix for embedding generation failures (see [ChromaDB Issue](2025-07-19_chromadb_issue.md))
- **Prevention Measures**: Added ChromaDB manager and safeguards to prevent future process conflicts (see [Prevention Measures](2025-07-19_prevent_chromadb_issues.md))

### Migration Notes
- **Database**: Existing papers automatically migrated to new schema
- **Vector Embeddings**: Complete regeneration required due to ChromaDB corruption
- **Backward Compatibility**: All existing functionality preserved
- **No Data Loss**: Paper database fully preserved, only embeddings needed regeneration

---

## Documentation

- [ChromaDB Issue (2025-07-19)](2025-07-19_chromadb_issue.md) - Details on vector database compatibility issues and resolution
- [Database Migration Summary (2025-07-19)](2025-07-19_database_migration_summary.md) - Complete database schema enhancement details  
- [Vector DB Reconstruction (2025-07-19)](2025-07-19_vector_db_reconstruction.md) - Vector database corruption fix and rebuild process
- [Prevention Measures (2025-07-19)](2025-07-19_prevent_chromadb_issues.md) - Code changes and safeguards to prevent future ChromaDB issues