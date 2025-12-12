# etl.py
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dateutil.parser import parse as parse_date

NORMALIZED_COLS = [
    "date",          # datetime64
    "merchant_raw",  # str
    "amount",        # float
    "currency",      # str
    "account_id",    # str
    "source_file",   # str
    "category",      # str 
]


def normalize_raw_df(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    desc_col: str = "Description",
    amount_col: str = "Amount",
    currency: str = "USD",
    account_id: str = "default",
    source_file: str = "unknown.csv",
) -> pd.DataFrame:
    # Normalize a raw bank/CC dataframe into the common schema
    for col in [date_col, desc_col, amount_col]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in raw dataframe. Got columns: {df.columns.tolist()}"
            )

    # Normalize date
    df_norm = pd.DataFrame()
    df_norm["date"] = df[date_col].apply(lambda d: parse_date(str(d)))

    # Description
    df_norm["merchant_raw"] = df[desc_col].astype(str).str.strip()

    # Amount as float
    df_norm["amount"] = df[amount_col].astype(float)

    # Meta
    df_norm["currency"] = currency
    df_norm["account_id"] = account_id
    df_norm["source_file"] = source_file

    # Category if present, else NaN
    if "category" in df.columns:
        df_norm["category"] = df["category"].astype(str)
    else:
        df_norm["category"] = pd.NA

    return df_norm[NORMALIZED_COLS].copy()


def normalize_raw_csv(
    csv_path: Path,
    *,
    date_col: str = "Date",
    desc_col: str = "Description",
    amount_col: str = "Amount",
    currency: str = "USD",
    account_id: str = "default",
) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    return normalize_raw_df(
        df_raw,
        date_col=date_col,
        desc_col=desc_col,
        amount_col=amount_col,
        currency=currency,
        account_id=account_id,
        source_file=csv_path.name,
    )


def save_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"transactions_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"[ETL] Saved normalized transactions to {out_path}")
    return out_path


def interactive_labeling(df: pd.DataFrame, max_to_label: int = 50) -> pd.DataFrame:
    # CLI labeling loop for a small seed set
    to_label = df[df["category"].isna()].copy()
    if to_label.empty:
        print("[ETL] No unlabeled transactions found.")
        return df

    print(f"[ETL] Found {len(to_label)} unlabeled rows. Labeling up to {max_to_label}.")
    print("Suggested categories: food, rent, utilities, entertainment, transportation, income, other")
    print("Type 'skip' to skip a row, or 'quit' to stop.")

    labeled = 0
    for idx, row in to_label.head(max_to_label).iterrows():
        print("-" * 60)
        print(f"Date:    {row['date'].date()}")
        print(f"Desc:    {row['merchant_raw']}")
        print(f"Amount:  {row['amount']:.2f}")

        cat = input("Enter category: ").strip()
        if cat.lower() == "quit":
            break
        if cat.lower() == "skip" or not cat:
            continue

        df.at[idx, "category"] = cat
        labeled += 1

    print(f"[ETL] Labeled {labeled} transactions.")
    return df


def main():
    ap = argparse.ArgumentParser(description="Normalize raw bank/CC CSV and optionally label categories.")
    ap.add_argument("csv_path", help="Path to raw CSV")
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--desc-col", default="Description")
    ap.add_argument("--amount-col", default="Amount")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--account-id", default="default")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--label", action="store_true", help="Interactively label a seed set of categories")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    df = normalize_raw_csv(
        csv_path,
        date_col=args.date_col,
        desc_col=args.desc_col,
        amount_col=args.amount_col,
        currency=args.currency,
        account_id=args.account_id,
    )

    if args.label:
        df = interactive_labeling(df)

    save_processed(df, Path(args.processed_dir))


if __name__ == "__main__":
    main()
