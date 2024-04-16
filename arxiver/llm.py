from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def summarize_summary(summary):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert summarizer capable of distilling complex information into its essence. Your summaries should be concise, informative, and limited to two sentences.",
            },
            {
                "role": "user",
                "content": f"Summarize this text in two sentences: {summary}",
            },
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content


def choose_summaries(summaries, k):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert summarizer capable of distilling complex information into its essence and a skilled evaluator of cutting edge ideas. Your choices should be based on the most interesting, novel, and cutting edge ideas.",
            },
            {
                "role": "user",
                "content": f"From the following article summaries, pick the {k} most interesting, novel, and cutting edge ideas and return a json list with 'id' and 'summary' for each.  You may also include a 'reason' for each choice.: {summaries}",
            },
        ],
        max_tokens=3000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip("json\n").strip("`").strip()
