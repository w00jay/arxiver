import json

import requests
import streamlit as st

st.title("Arxiver Console")

api_url = "http://127.0.0.1:8000"


# Function to handle POST requests
def post_request(endpoint, data):
    response = requests.post(f"{api_url}/{endpoint}", json=data)
    return response.text


# Function to handle GET requests
def get_request(endpoint, data):
    response = requests.get(f"{api_url}/{endpoint}", json=data)
    return response.text


# Ingest
with st.form("ingest"):
    days = st.number_input("Enter number of days for ingestion:", min_value=1, value=2)
    submit_ingest = st.form_submit_button("Ingest Articles")
    if submit_ingest:
        response = post_request("ingest", {"days": days})
        st.json(response)

# Embed
with st.form("embed"):
    days_embed = st.number_input(
        "Enter number of days for embedding:", min_value=1, value=2
    )
    submit_embed = st.form_submit_button("Create Embeddings")
    if submit_embed:
        response = post_request("embed", {"days": days_embed})
        st.json(response)

# Query
with st.form("query"):
    query_text = st.text_input("Enter query text:")
    submit_query = st.form_submit_button("Query Articles")
    if submit_query:
        response = post_request("query", {"query_text": query_text})
        st.json(response)

# Choose
with st.form("choose"):
    choose_text = st.text_input("Enter choose query text:")
    i = st.number_input("Select top i summaries:", min_value=1, value=2)
    k = st.number_input("From top k articles:", min_value=1, value=10)
    submit_choose = st.form_submit_button("Choose Articles")
    if submit_choose:
        response = post_request("choose", {"query_text": choose_text, "i": i, "k": k})
        st.json(response)

# Recommend
with st.form("recommend"):
    days_back = st.number_input(
        "Enter number of days back for recommendation:", min_value=1, value=1
    )
    submit_recommend = st.form_submit_button("Get Recommendations")
    if submit_recommend:
        response = get_request("recommend", {"days_back": days_back})
        st.json(response)

# Fill Missing Embeddings
st.subheader("Fill Missing Embeddings")
if st.button("Fill Missing Embeddings"):
    response = get_request("fill-missing-embeddings", {})
    st.json(response)

# Import Article
with st.form("import_article"):
    arxiv_id = st.text_input("Enter arXiv ID to import:")
    submit_import = st.form_submit_button("Import Article")
    if submit_import:
        response = post_request("import", {"arxiv_id": arxiv_id})
        st.json(response)

# Summarize
with st.form("summarize"):
    paper_id = st.text_input("Enter paper ID:")
    submit_summarize = st.form_submit_button("Summarize Article")
    if submit_summarize:
        response = post_request("summarize", {"paper_id": paper_id})
        st.json(response)
