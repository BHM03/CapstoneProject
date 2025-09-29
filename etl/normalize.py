import hashlib, re
import pandas as pd
from datetime import datetime

MERCHANT_PATTERNS = [
    (r"\s+POS\s+PURCHASE.*", ""), # POS Purchase
    (r"\s+#\d{3,}", ""), #1233
    (r"(?i)\bONLINE TRANSFER\b.*", "ONLINE TRANSFER"), #onLIne Transfer
    (r"\s{2,}", " ") 
]

def normalize_merchant(s: str) -> str:
    if not isinstance(s, str): return ""
    x = s.strip().upper()
    for pat, repl in MERCHANT_PATTERNS:
        x = re.sub(pat, repl, x)
    return x.strip()

def to_cents(amount):
    if pd.isna(amount): return 0
    return int(round(float(amount) * 100))

def make_txn_id(account_id, date_str, amount_cents, merchant_norm, raw_id):
    key = f"{account_id}|{date_str}|{amount_cents}|{merchant_norm}|{raw_id or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def parse_date(s, fmt):
    return pd.to_datetime(s, format=fmt, errors="coerce").dt.date

def dedupe(df):
    df = df.sort_values(["is_pending", "source_file"]).drop_duplicates(
        subset=["txn_id"], keep="first"
    )
    return df

def sign_convention(amount_cents, expense_negative=True):
    return amount_cents if expense_negative else -amount_cents
