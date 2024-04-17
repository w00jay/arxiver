import json
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
    try:
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
            temperature=0.0,
        )
        print(response.choices[0].message.content)
        response_content = response.choices[0].message.content.strip("`").strip().removeprefix("json\n")
        
        # Debugging: Print the raw response content
        print("Raw response content:", response_content)
        
        # Ensure the response is not empty before parsing
        if response_content:
            parsed_response = json.loads(response_content)
            return parsed_response
        else:
            print("Response content is empty.")
            return []

    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return []
    except Exception as e:
        print("An error occurred:", e)
        return []