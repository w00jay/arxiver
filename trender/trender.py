import sqlite3
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def query_database():
    # Connect to the SQLite database
    conn = sqlite3.connect("../data/arxiv_papers.db")
    cursor = conn.cursor()

    # Query all entries in the 'papers' table
    cursor.execute("SELECT paper_id, title, summary, updated FROM papers")
    rows = cursor.fetchall()

    # Construct the list of dictionaries
    articles = [
        {"id": row[0], "title": row[1], "summary": row[2], "date": row[3]}
        for row in rows
    ]

    # Close the database connection
    conn.close()

    return articles


# articles = [
#     {'id': 1, 'title': 'Deep Learning for AI', 'summary': 'Deep learning transforms...', 'date': '2023-01-01'},
#     # Add more articles here
# ]
articles = query_database()

# Convert articles to a DataFrame
df = pd.DataFrame(articles)

# Preprocess summaries: tokenize, lower, remove stopwords
stop_words = set(stopwords.words("english"))
df["processed_summary"] = df["summary"].apply(
    lambda x: [
        word.lower()
        for word in word_tokenize(x)
        if word.isalpha() and word.lower() not in stop_words
    ]
)

# Frequency analysis
all_words = [word for summary in df["processed_summary"] for word in summary]
word_freq = Counter(all_words)

# Convert to DataFrame for easier manipulation
freq_df = pd.DataFrame(word_freq.items(), columns=["word", "frequency"]).sort_values(
    by="frequency", ascending=False
)

# Trend analysis: Count word occurrences by year
df["year"] = pd.to_datetime(df["date"]).dt.year
trends = (
    df.explode("processed_summary")
    .groupby(["year", "processed_summary"])
    .size()
    .reset_index(name="counts")
)

# Saving results to SQLite database
conn = sqlite3.connect("../data/arxiv_analysis.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS word_frequency (
    word TEXT PRIMARY KEY,
    frequency INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS word_trends (
    year INTEGER,
    word TEXT,
    count INTEGER,
    PRIMARY KEY (year, word)
)
""")

# Insert frequency analysis results
for _, row in freq_df.iterrows():
    cursor.execute(
        "REPLACE INTO word_frequency (word, frequency) VALUES (?, ?)",
        (row["word"], row["frequency"]),
    )

# Insert trend analysis results
for _, row in trends.iterrows():
    cursor.execute(
        "REPLACE INTO word_trends (year, word, count) VALUES (?, ?, ?)",
        (row["year"], row["processed_summary"], row["counts"]),
    )

conn.commit()
conn.close()

print("Frequency and trend analysis data saved to database.")

# Visualization

# Connect to the SQLite database
conn = sqlite3.connect("../data/arxiv_analysis.db")

# Load the data from the database
word_freq = pd.read_sql_query(
    "SELECT * FROM word_frequency ORDER BY frequency DESC LIMIT 20", conn
)
word_trends = pd.read_sql_query(
    "SELECT * FROM word_trends WHERE word IN (SELECT word FROM word_frequency ORDER BY frequency DESC LIMIT 10)",
    conn,
)

conn.close()

# Visualization
plt.figure(figsize=(10, 8))

# Top 20 most frequent words visualization
sns.barplot(x="frequency", y="word", data=word_freq, palette="viridis")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

# Trends of top 10 most frequent words over years
plt.figure(figsize=(12, 8))
for word in word_trends["word"].unique():
    subset = word_trends[word_trends["word"] == word]
    plt.plot(subset["year"], subset["count"], marker="o", label=word)

plt.title("Trend Analysis of Top 10 Words Over Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(title="Word")
plt.tight_layout()
plt.show()
