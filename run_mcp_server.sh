#!/bin/bash
# Script to run the Arxiver MCP server with proper PYTHONPATH

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec python arxiver/mcp_server.py "$@"