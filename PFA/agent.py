import json
import os
import re

import pandas as pd
from openai import OpenAI

# strip <think>...</think> tags
def strip_think_tags(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)

    lower = text.lower()
    start = lower.find("<think>")
    end = lower.find("</think>")

    # only strip if it sees both tags in the right order
    if start != -1 and end != -1 and end > start:
        before = text[:start]
        after = text[end + len("</think>") :]
        return (before + after).strip()

    # no well formed block found
    return text.strip()



# Hugging Face Router config

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_BASE_URL = "https://router.huggingface.co/v1"

# Chat capable model served via HF Inference provider
HF_MODEL_ID = "HuggingFaceTB/SmolLM3-3B:hf-inference"


def _get_hf_client() -> OpenAI | None:
    
    # Build an OpenAI compatible client that actually talks to Hugging Face Router.
    
    if not HF_TOKEN:
        return None

    client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=HF_TOKEN,
    )
    return client


# Build FULL dataset payload for the LLM
def build_full_data_payload(df: pd.DataFrame) -> dict:
    """
    Build a JSON serializable representation of the full dataset

    - All transactions
    - Category totals
    - Overall date range
    - Number of transactions
    """
    if df.empty:
        return {
            "n_transactions": 0,
            "transactions": [],
            "categories": {},
            "date_range": None,
        }

    df = df.copy()

    # ensure date is a string in YYYY-MM-DD format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Ensure category exists
    if "category" not in df.columns:
        df["category"] = "unknown"

    # Try to ensure amount is float for nicer output
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Compute category totals
    if "amount" in df.columns:
        totals = df.groupby("category")["amount"].sum().to_dict()
    else:
        totals = {}

    # Build date range
    date_range = None
    if "date" in df.columns:
        # some rows might have NaT/ NaT if parsing failed
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.notna().any():
            start = dates.min().date().isoformat()
            end = dates.max().date().isoformat()
            date_range = {"start": start, "end": end}

    # Convert entire DataFrame to records for the model
    transactions = df.to_dict(orient="records")

    payload = {
        "n_transactions": len(df),
        "transactions": transactions,
        "categories": totals,
        "date_range": date_range,
    }
    return payload


# answer_question_llm
def answer_question_llm(df: pd.DataFrame, question: str) -> str:
    """
    Use Hugging Face Router to answer a question about the user's finances
    based on the full transaction dataset

    The entire filtered DataFrame `df` is serialized and sent to the model
    as JSON, along with some helpful aggregates
    """
    if df.empty:
        return "I don't have any transactions yet, so I can't analyze your finances."

    client = _get_hf_client()
    if client is None:
        return (
            "The AI assistant is not configured yet.\n\n"
            "Set the HUGGINGFACEHUB_API_TOKEN environment variable to your Hugging Face token."
        )

    data_payload = build_full_data_payload(df)

    system_prompt = (
        "You are a careful, honest personal finance assistant. "
        "You are given the user's FULL transaction dataset as JSON. "
        "Each transaction includes fields like date, merchant_raw, amount, category, "
        "and account_id (when available). "
        "Use ONLY this data to answer questions. "
        "Treat negative amounts as expenses and positive amounts as income. "
        "Be concise and specific. When giving dollar amounts, round to 2 decimals. "
        "If you are unsure or the data does not contain enough information, "
        "say so honestly."
    )

    user_prompt = f"""
Here is the user's full transaction dataset (JSON):

{json.dumps(data_payload, indent=2, default=str)}

User question:
{question}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
        model=HF_MODEL_ID,
        messages=messages,
        max_tokens=768, 
        temperature=0.3,
        )

        choice = completion.choices[0]
        content = choice.message.content

        # see if the model hit the length limit
        print("HF finish_reason:", getattr(choice, "finish_reason", None))

        if not isinstance(content, str):
            content = str(content)

        content = strip_think_tags(content)
        return content.strip()

    except Exception as e:
        # Print detailed error to your terminal for debugging
        print("HF Router / OpenAI-compatible error in answer_question_llm:", repr(e))
        return f"The external AI assistant is unavailable right now: {type(e).__name__}: {e}"
