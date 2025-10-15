import argparse
from pathlib import Path
import pandas as pd

from normalize import(
    normalize_merchant, to_cents, make_txn_id, parse_date, dedupe, sign_convention
)

# command line argument parsing
def parse_args():
    ap = argparse.ArgumentParser(description="Ingest a bank/cc CSV into warehouse.")
    ap.add_argument("csv_path", help="Path to raw CSV to ingest")
    # map raw columns to fields
    ap.add_argument("--date-col", default="Date", help="Raw date column name")
    ap.add_argument("--desc-col", default="Description", help="Raw description/merchant column name")
    ap.add_argument("--amount-col", default="Amount", help="Raw amount column name")
    ap.add_argument("--pending-col", default=None, help="Raw pending column (optional)")
    ap.add_argument("--id-col", default=None, help="Raw transaction id column (optional)")
    ap.add_argument("--date-format", default="%m/%d/%Y", help="strftime format for --date-col")
    ap.add_argument("--account-id", default=None, help="Account ID to attach (e.g., 'chase:chk')")
    ap.add_argument("--currency", default="USD", help="Currency code")
    ap.add_argument("--expense-negative", action="store_true",
                    help="If set, assumes expenses should be negative (default behavior).")
    ap.add_argument("--data-dir", default="data", help="Base data directory")
    return ap.parse_args()

def main():
    args = parse_args()

    # make path objects for the input CSV and output folders
    csv_path = Path(args.csv_path).resolve()
    base_dir = Path(args.data_dir)
    # ensure output directories exist
    staging_dir = base_dir / "staging"
    warehouse_dir = base_dir / "warehouse"
    staging_dir.mkdir(parents=True, exist_ok=True)
    warehouse_dir.mkdir(parents=True, exist_ok=True)

    # if --account-id wasn’t passed, get a simple ID from the filename
    if args.account_id:
        account_id = args.account_id
    else:
        stem = csv_path.stem.lower()
        bank = stem.split("_")[0].split("-")[0]
        account_id = f"{bank}:csv"

    # Extract
    df_raw = pd.read_csv(csv_path)

    # sanity checks
    for col in [args.date_col, args.desc_col, args.amount_col]:
        if col not in df_raw.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path.name}")

    # Transform

    # new dataframe for normalized columns only
    tmp = pd.DataFrame()
    # parse raw date strings into real dates using helper function
    tmp["date"] = parse_date(df_raw[args.date_col], args.date_format)
    tmp["merchant_raw"] = df_raw[args.desc_col].astype(str)
    tmp["merchant_norm"] = tmp["merchant_raw"].map(normalize_merchant)

    # amount to integer cents
    amt = df_raw[args.amount_col].astype(float)
    tmp["amount_cents"] = (amt * 100).round().astype(int)
    tmp["amount_cents"] = tmp["amount_cents"].apply(
        lambda x: sign_convention(x, expense_negative=True)  # keep expenses negative, income positive
    )

    # pending flag
    if args.pending_col and args.pending_col in df_raw.columns:
        tmp["is_pending"] = (
            df_raw[args.pending_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "yes", "pending"])
        )
    else:
        tmp["is_pending"] = False

    # raw_id
    if args.id_col and args.id_col in df_raw.columns:
        tmp["raw_id"] = df_raw[args.id_col].astype(str)
    else:
        tmp["raw_id"] = ""

    # add metadata columns
    tmp["currency"] = args.currency
    tmp["source_file"] = csv_path.name
    tmp["account_id"] = account_id

    # build a stable transaction ID
    tmp["txn_id"] = tmp.apply(
        lambda r: make_txn_id(
            r["account_id"],
            str(r["date"]) if pd.notna(r["date"]) else "",
            int(r["amount_cents"]),
            r["merchant_norm"],
            r["raw_id"],
        ),
        axis=1,
    )

    # drop bad dates
    tmp = tmp[tmp["date"].notna()].copy()

    # initialize labels 
    if "label" not in tmp.columns:
        tmp["label"] = pd.NA
    if "category_rule" not in tmp.columns:
        tmp["category_rule"] = pd.NA

    # Load
    
    # write staging
    stage_file = staging_dir / f"{csv_path.stem}.csv"
    tmp.to_csv(stage_file, index=False)

    # load into warehouse
    wh_file = warehouse_dir / "transactions.csv"
    if wh_file.exists():
        wh = pd.read_csv(wh_file, dtype={"raw_id": str})
        merged = pd.concat([wh, tmp], ignore_index=True)
    else:
        merged = tmp

    # prefer posted over pending. simple "latest file wins" via source_file ordering
    merged = merged.sort_values(["is_pending", "source_file"], ascending=[True, True])
    merged = merged.drop_duplicates(subset=["txn_id"], keep="first")

    # save back
    merged.to_csv(wh_file, index=False)

    print(f"Ingested {len(tmp)} rows from {csv_path.name}")
    print(f"Warehouse now has {len(merged)} rows -> {wh_file}")

# scripts runs only if the file is executed
if __name__ == "__main__":
    main()
