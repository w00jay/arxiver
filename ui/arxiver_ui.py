import json
from datetime import datetime

# import logging
import requests
import streamlit as st

st.title("Arxiver Console")

api_url = "http://127.0.0.1:8000"

# logging.basicConfig(level=logging.INFO)


# Function to handle POST requests
def post_request(endpoint, data):
    response = requests.post(f"{api_url}/{endpoint}", json=data)
    return response.text


# Function to handle GET requests
def get_request(endpoint, data):
    response = requests.get(f"{api_url}/{endpoint}", json=data)
    return response.text


tab_query, tab_ingest = st.tabs(["Query", "Ingest"])

with tab_ingest:
    # Ingest
    with st.form("ingest"):
        days = st.number_input(
            "Enter number of days to look back beginning today:", min_value=1, value=2
        )
        submit_ingest = st.form_submit_button("Ingest Articles")
        if submit_ingest:
            response = post_request(
                "ingest",
                {
                    "start_date": datetime.now().date().strftime("%Y-%m-%d"),
                    "days": days,
                },
            )
            st.json(response)

    # Ingest_from_date
    with st.form("ingest_from_date"):
        start_date = st.date_input(
            "Select the starting date for ingestion:", value=datetime.now().date()
        )
        days = st.number_input(
            "Enter number of days to look back:", min_value=1, value=2
        )
        submit_ingest = st.form_submit_button("Ingest Articles")
        if submit_ingest:
            response = post_request(
                "ingest", {"start_date": start_date.strftime("%Y-%m-%d"), "days": days}
            )
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

    # Import Article
    with st.form("import_article"):
        arxiv_id = st.text_input("Enter arXiv ID to import:")
        submit_import = st.form_submit_button("Import Article")
        if submit_import:
            response = post_request("import", {"arxiv_id": arxiv_id})
            st.json(response)

    # Fill Missing Article Summaries
    with st.container():
        st.subheader("Fill Missing Article Summaries")
        if st.button("Fill Missing Article Summaries"):
            response = get_request("fill-missing-summaries", {})
            st.json(response)

    # Fill Missing Embeddings
    with st.container():
        st.subheader("Fill Missing Embeddings")
        if st.button("Fill Missing Embeddings"):
            response = get_request("fill-missing-embeddings", {})
            st.json(response)


with tab_query:
    # Recommend
    with st.form("recommend"):
        days_back = st.number_input(
            "Enter number of days back for recommendation:", min_value=1, value=1
        )
        submit_recommend = st.form_submit_button("Get Recommendations")
        if submit_recommend:
            response = get_request("recommend", {"days_back": days_back})
            recommendations = json.loads(response)
            # logging.info(recommendations)

            c = st.container()
            if not recommendations:
                c.write("No recommendations were returned.")
            else:
                for item in recommendations:
                    # logging.info(item)
                    if isinstance(item, dict):
                        pdf_url = f"{item['id'].replace('abs', 'pdf')}"
                        ex = c.expander(f"**{item['title']}**")
                        ex.write(f"{item['summary']}")
                        ex.link_button(
                            "Paper Summary", item["id"], use_container_width=True
                        )
                        ex.link_button("PDF", pdf_url, use_container_width=True)
                    else:
                        st.warning(f"Unexpected item format: {item}")

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
            response = post_request(
                "choose", {"query_text": choose_text, "i": i, "k": k}
            )
            st.json(response)

    # Summarize
    with st.form("summarize"):
        paper_id = st.text_input(
            "Enter full arXiv URL (e.g., http://arxiv.org/abs/2404.08836v1):"
        )
        submit_summarize = st.form_submit_button("Summarize Article")
        if submit_summarize:
            response = post_request("summarize", {"paper_id": paper_id})
            st.json(response)
