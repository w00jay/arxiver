## arxiver

`arxiver` is a set of tools designed to manage and search arXiv articles of interest.


### __Status: Pre-release hack__

### tl;dr
```bash
# 1. Start the server in /arxiver

> cd arxiver && python main.py --help
...
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  add-interested-column  Add an 'interested' column to the papers table.
  ingest                 Performs the ingestion process directly via CLI,...
  webserver              Starts the FastAPI web server.

> python main.py webserver
...
INFO:     Started server process [1085226]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Or use uvicorn:
```bash
uvicorn main:app --reload --port 8000
```


```bash
# 2. In another terminal, start the Streamlit UI in /ui,

> cd ../ui && streamlit run arxiver_ui.py --server.port 8001

  You can now view your Streamlit app in your browser.

  URL: http://localhost:8001

# OR make HTTP requests to the server:

> curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"days": 2}'

{"message":"Ingestion process started for the past 2 days."}
```

