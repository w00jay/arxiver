import json
import logging

from dotenv import load_dotenv

try:
    from .logger import setup_logging
except ImportError:
    from logger import setup_logging
from openai import OpenAI

load_dotenv()
client = OpenAI()

setup_logging()


def summarize_summary(summary, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # "gpt-3.5-turbo",
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
            temperature=0.0,
        )
        logging.debug(response.choices[0].message.content)
        response_content = (
            response.choices[0]
            .message.content.strip("`")
            .strip()
            .removeprefix("json\n")
        )

        # Debugging
        # logging.debug("Raw response content:", response_content)

        if response_content:
            parsed_response = json.loads(response_content)
            return parsed_response
        else:
            logging.debug("Response content is empty.")
            return []

    except json.JSONDecodeError as e:
        logging.error("Failed to decode JSON:", e)
        return []
    except Exception as e:
        logging.error("An error occurred:", e)
        return []
