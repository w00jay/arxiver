import pytest
from llm import summarize_summary
from openai import OpenAI


@pytest.fixture
def sample_text():
    return """Large language models (LLMs) have revolutionized natural language processing by demonstrating remarkable capabilities in understanding and generating human-like text. These models, trained on vast amounts of data, can perform tasks ranging from text summarization to code generation. However, their large size and computational requirements pose challenges for deployment and efficiency."""


def test_summarize_summary_comparison(sample_text):
    # Test GPT-4o-mini
    gpt4_mini_result = summarize_summary(sample_text, model="gpt-4o-mini")

    # Test GPT-3.5-turbo
    gpt3_turbo_result = summarize_summary(sample_text, model="gpt-3.5-turbo")

    # Basic assertions
    assert len(gpt4_mini_result) > 0
    assert len(gpt3_turbo_result) > 0

    # Compare lengths (GPT-4o-mini should be more concise)
    assert len(gpt4_mini_result) <= len(gpt3_turbo_result)

    # Compare content similarity (should be similar but not identical)
    similarity = len(
        set(gpt4_mini_result.split()) & set(gpt3_turbo_result.split())
    ) / max(len(gpt4_mini_result.split()), len(gpt3_turbo_result.split()))
    assert 0.3 < similarity < 0.8  # Expect some overlap but not identical
