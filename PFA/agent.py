# agent.py
import json
import os

import pandas as pd
from openai import OpenAI

import re

def strip_think_tags(text: str)-> str:

    # remove ANY <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# Hugging Face token from env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# HF router base URL 
HF_BASE_URL = "https://router.huggingface.co/v1"

# Chat capable model served via HF Inference provider
HF_MODEL_ID = "HuggingFaceTB/SmolLM3-3B:hf-inference"


def _get_hf_client()-> OpenAI | None:
    
    # build an OpenAI-compatible client that actually talks to Hugging Face Router
    if not HF_TOKEN:
        return None

    client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=HF_TOKEN,
    )
    return client

def build_summary_for_llm(df: pd.DataFrame) -> dict:
    
    # build a compact summary for the LLM; category totals, recent transactions, date range
    
    if df.empty:
        return {"categories": {}, "recent": [], "date_range": None}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    totals = df.groupby("category")["amount"].sum().to_dict()

    recent = (
        df.sort_values("date", ascending=False)
          .head(20)
          .loc[:, ["date", "merchant_raw", "amount", "category"]]
          .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
          .to_dict(orient="records")
    )

    summary = {
        "categories": totals,
        "recent": recent,
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
    }
    return summary


def answer_question_llm(df: pd.DataFrame, question: str) -> str:
    """
    Use Hugging Face Router to answer a question
    about the user's finances based on categorized transactions
    """
    if df.empty:
        return "I don't have any transactions yet, so I can't analyze your finances."

    client = _get_hf_client()
    if client is None:
        return (
            "The AI assistant is not configured yet.\n\n"
            "Set the HUGGINGFACEHUB_API_TOKEN environment variable to your Hugging Face token."
        )

    summary = build_summary_for_llm(df)

    system_prompt = (
        "You are a careful, honest personal finance assistant. "
        "Use ONLY the provided spending summary to answer questions. "
        "Be concise and specific. When giving dollar amounts, round to 2 decimals. "
        "If you are unsure, say so honestly."
    )

    user_prompt = f"""
Here is the user's financial summary (JSON):

{json.dumps(summary, indent=2, default=str)}

User question:
{question}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        content = completion.choices[0].message.content

        # convert to str
        if not isinstance(content, str):
            content = str(content)

        # strip <think> tags
        content = strip_think_tags(content)

        return content.strip()
    except Exception as e:
        # print detailed error to terminal for debugging
        print("HF Router / OpenAI-compatible error in answer_question_llm:", repr(e))
        return f"The external AI assistant is unavailable right now: {type(e).__name__}: {e}"
