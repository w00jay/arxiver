# Fix: Prevent Duplicate LLM Calls During Re-ingestion

**Date**: 2025-07-19
**Issue**: When ingesting the same date twice, the system was generating duplicate concise summaries for papers that already had them.

## Root Cause
**Critical Issue**: The `insert_article()` function was using `INSERT OR REPLACE` which **destroyed existing concise summaries** during re-ingestion:

1. `fetch_articles_for_date()` retrieves the same articles from arXiv API
2. `insert_article()` uses `INSERT OR REPLACE` but only specifies certain fields (NOT `concise_summary`)
3. When replacing existing rows, `concise_summary` gets reset to NULL, wiping out existing summaries
4. `summarize_article()` then processes these papers because they appear to have no summaries

Secondary issue: The `ingest_process` function was calling `summarize_article()` for all returned paper IDs without checking if they already had summaries.

## Solution
**Primary Fix**: Changed `insert_article()` from `INSERT OR REPLACE` to `INSERT OR IGNORE` to preserve existing data:

```sql
-- Before: INSERT OR REPLACE INTO papers(...) 
-- After: INSERT OR IGNORE INTO papers(...)
```

This ensures that existing papers with concise summaries are left completely untouched during re-ingestion.

**Secondary Fix**: Modified `ingest_process()` function to check if papers already have concise summaries before calling `summarize_article()`:

```python
# Generate concise summaries for articles that don't have them yet
for article_id in article_ids:
    paper = get_paper_by_id(conn, article_id)
    if paper and (not paper[3] or paper[3].strip() == ""):  # Check concise_summary field
        summarize_article(PAPERS_DB, article_id)
    else:
        logger.info(f"Skipping {article_id} - concise summary already exists.")
```

Also improved the summary existence check in multiple functions to handle both `NULL` and empty string cases:

```python
# Before: if not concise_summary:
# After: if not concise_summary or concise_summary.strip() == "":
```

## Files Modified
- `arxiver/database.py:79` - **CRITICAL**: Changed `INSERT OR REPLACE` to `INSERT OR IGNORE`
- `arxiver/main.py:107` - Updated condition in `summarize_article()`
- `arxiver/main.py:82` - Updated condition in `summarize_recent_entries()`  
- `arxiver/main.py:149-163` - Added pre-check and detailed logging in `ingest_process()`
- `arxiver/main.py:373` - Updated condition in `/summarize` endpoint

## Testing
- Verified that papers with existing summaries are properly skipped
- Confirmed that papers without summaries still get processed
- Tested the logic on actual database records

## Result  
- ✅ Eliminates duplicate LLM calls during re-ingestion
- ✅ Preserves existing functionality for new papers
- ✅ Provides clear logging when skipping papers with summaries
- ✅ Handles both NULL and empty string concise_summary values