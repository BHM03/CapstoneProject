# app.py
from pathlib import Path
from typing import Optional

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from etl import normalize_raw_df, save_processed
from agent import answer_question_llm

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODEL_PATH = MODELS_DIR / "category_classifier.joblib"


# Streamlit page config
st.set_page_config(
    page_title="Personal Finance Analyzer (ML + AI)",
    layout="wide",
)



# Caching helpers
@st.cache_data
def load_transactions() -> pd.DataFrame:
    
    # Load all processed transactions from data/processed
    # Excludes any files with 'training' in the name so the UI only
    # reflects your real ingested data, not synthetic training data
    files = sorted(PROCESSED_DIR.glob("transactions_*.csv"))
    files = [f for f in files if "training" not in f.name]

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Basic sanity: ensure expected columns exist
    if "account_id" not in df.columns:
        df["account_id"] = "default"

    if "category" not in df.columns:
        df["category"] = "unknown"

    # Make sure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_resource
def load_classifier():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None



# ML and analytics helpers
def ensure_categories_with_classifier(df: pd.DataFrame) -> pd.DataFrame:
    
    #Use the trained classifier to categorize any missing/unknown categories
    #This is where your ML model is actually used in the app
    clf = load_classifier()
    if clf is None:
        st.warning("No trained classifier found. Run `python3 train_classifier.py` first.")
        if "category" not in df.columns:
            df["category"] = "unknown"
        return df

    # Ensure the category column exists
    if "category" not in df.columns:
        df["category"] = pd.NA

    # Make sure category is a string dtype before using .str
    df["category"] = df["category"].astype("string")

    # Find rows that need prediction 
    mask = df["category"].isna() | df["category"].str.lower().eq("unknown")
    to_predict = df[mask].copy()

    if to_predict.empty:
        return df

    # Safety to make sure merchant_raw exists
    if "merchant_raw" not in to_predict.columns:
        st.error("Missing 'merchant_raw' column; cannot classify transactions.")
        return df

    # Predict categories from merchant_raw using trained model
    preds = clf.predict(to_predict["merchant_raw"])
    df.loc[mask, "category"] = preds

    return df


def compute_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(["year_month", "category"])["amount"]
          .sum()
          .reset_index()
    )
    return monthly


def compute_forecast(monthly: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    
    # Simple per-category forecast: next month = mean of last N months
    if monthly.empty:
        return pd.DataFrame()

    last_month = monthly["year_month"].max()
    next_month = last_month + pd.offsets.MonthBegin(1)

    rows = []
    for cat, grp in monthly.groupby("category"):
        tail = grp.sort_values("year_month").tail(window)
        if len(tail) == 0:
            continue
        rows.append({
            "year_month": next_month,
            "category": cat,
            "amount_forecast": tail["amount"].mean(),
        })

    return pd.DataFrame(rows)


def detect_subscriptions(df: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    
    #Heuristic: merchants that appear in >= min_months distinct months are likely subscriptions
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    stats = (
        df.groupby("merchant_raw")
          .agg(
              months=("year_month", "nunique"),
              n_txns=("amount", "size"),
              avg_amount=("amount", "mean"),
              std_amount=("amount", "std"),
          )
          .reset_index()
    )
    subs = stats[stats["months"] >= min_months].copy()
    subs = subs.sort_values("avg_amount", ascending=False)
    return subs


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    
    #Aggregate net spending by month (in dollars)
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month", as_index=False)["amount"].sum()
    return monthly


def simple_forecast_next_month(monthly_df: pd.DataFrame) -> Optional[float]:
    
    # Very naive forecast: mean of last 3 months (in dollars)
    
    if monthly_df.empty:
        return None
    last_n = monthly_df.tail(3)["amount"]
    return float(last_n.mean())


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"



# Main app
def main():
    st.title("Personal Finance Analyzer - ML Classifier + AI Agent")
    st.caption("Upload bank CSVs, categorize with your ML model, explore dashboards, and ask an AI about your spending.")

    # Sidebar, account profile, and CSV upload
    st.sidebar.header("Account profile")
    name = st.sidebar.text_input("Name", value=st.session_state.get("name", ""))
    monthly_budget = st.sidebar.number_input(
        "Monthly budget (optional)", min_value=0.0, value=0.0, step=100.0
    )
    st.session_state["name"] = name
    st.session_state["monthly_budget"] = monthly_budget

    st.sidebar.markdown("---")
    st.sidebar.header("Upload bank statements")
    upload = st.sidebar.file_uploader("Upload a raw CSV (Date, Description, Amount)", type=["csv"])
    account_id = st.sidebar.text_input("Account ID for this upload", value="default")

    if st.sidebar.button("Ingest uploaded CSV"):
        if upload is None:
            st.sidebar.warning("Please choose a CSV first.")
        else:
            raw_df = pd.read_csv(upload)
            try:
                norm_df = normalize_raw_df(
                    raw_df,
                    date_col="Date",
                    desc_col="Description",
                    amount_col="Amount",
                    currency="USD",
                    account_id=account_id,
                    source_file=upload.name,
                )
            except Exception as e:
                st.sidebar.error(
                    f"Failed to normalize. Make sure columns Date/Description/Amount exist. Error: {e}"
                )
            else:
                save_processed(norm_df, PROCESSED_DIR)
                st.sidebar.success("Uploaded and normalized successfully. Reload page to include new data.")
                # Clear cache so new data shows up
                load_transactions.clear()

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Transactions", "Categories", "Forecast", "Subscriptions", "Unknown", "AI Assistant"],
    )

    # Load and prepare data
    df = load_transactions()
    if df.empty:
        st.info("No transactions yet. Run `etl.py` or upload a CSV in the sidebar.")
        return

    # Ensure categories via ML classifier
    df = ensure_categories_with_classifier(df)

    # Intro
    if page != "AI Assistant":
        st.subheader("Intro")
        intro_text = (
            "This app uses your own trained machine learning classifier to categorize transactions, "
            "and an AI assistant to answer natural-language questions about your finances."
        )
        if name:
            intro_text = f"Hi {name}! " + intro_text
        st.write(intro_text)

    #  Global filters
    if page == "AI Assistant":
        df_filtered = df.copy()
    else:
        with st.expander("Global filters", expanded=True):
            col1, col2, col3 = st.columns(3)

            min_date = df["date"].min().date()
            max_date = df["date"].max().date()

            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                )

            with col2:
                end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                )

            with col3:
                accounts = sorted(df["account_id"].astype(str).unique())
                account_filter = st.multiselect(
                    "Accounts",
                    options=accounts,
                    default=accounts,
                )

        mask = (
            (df["date"] >= pd.to_datetime(start_date))
            & (df["date"] <= pd.to_datetime(end_date))
            & (df["account_id"].astype(str).isin(account_filter))
        )
        df_filtered = df[mask].copy()

    # Page: Overview
    if page == "Overview":
        st.subheader("Overview")

        if df_filtered.empty:
            st.warning("No transactions in the selected range.")
        else:
            total_spent = df_filtered["amount"].sum()
            n_txns = len(df_filtered)
            avg_txn = df_filtered["amount"].mean()

            first_date = df_filtered["date"].min().date()
            last_date = df_filtered["date"].max().date()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total net amount", format_currency(total_spent))
            with col2:
                st.metric("# of transactions", f"{n_txns}")
            with col3:
                st.metric("Avg per transaction", format_currency(avg_txn))
            with col4:
                st.metric("Date range", f"{first_date} → {last_date}")

            st.markdown("### Daily net spending")
            daily = df_filtered.copy()
            daily["day"] = daily["date"].dt.date
            daily_agg = daily.groupby("day", as_index=False)["amount"].sum()

            chart = (
                alt.Chart(daily_agg)
                .mark_line(point=True)
                .encode(
                    x="day:T",
                    y=alt.Y("amount:Q", title="Net amount (dollars)"),
                    tooltip=["day:T", "amount:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### Monthly spending by category (line chart for one category)")
            monthly = compute_monthly_totals(df_filtered)
            if not monthly.empty:
                cats = sorted(monthly["category"].dropna().unique().tolist())
                selected_cat = st.selectbox("Category", options=cats)
                monthly_cat = monthly[monthly["category"] == selected_cat]
                if not monthly_cat.empty:
                    line = (
                        alt.Chart(monthly_cat)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("year_month:T", title="Month"),
                            y=alt.Y("amount:Q", title="Amount (dollars)"),
                            tooltip=["year_month:T", "amount:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.write("No data for that category yet.")
            else:
                st.write("No monthly data yet.")

    # Page: Transactions
    elif page == "Transactions":
        st.subheader("Transactions")

        if df_filtered.empty:
            st.warning("No transactions in the selected range.")
        else:
            cats = sorted(df_filtered["category"].dropna().unique())
            selected_categories = st.multiselect(
                "Filter by category",
                options=cats,
                default=cats,
            )

            tx_mask = df_filtered["category"].isin(selected_categories)
            tx = df_filtered[tx_mask].copy()

            if tx.empty:
                st.warning("No transactions match the current filters.")
            else:
                st.write(f"Showing **{len(tx)}** transactions.")

                display_cols = [
                    "date",
                    "account_id",
                    "merchant_raw",
                    "category",
                    "amount",
                ]
                tx_display = tx[display_cols].copy()
                tx_display["amount"] = tx_display["amount"].apply(format_currency)

                st.dataframe(
                    tx_display.sort_values("date", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

    # Page: Categories
    elif page == "Categories":
        st.subheader("Spending by category")

        if df_filtered.empty:
            st.warning("No data in selected range.")
        else:
            cat_agg = (
                df_filtered.groupby("category", as_index=False)["amount"].sum()
            )

            st.markdown("### Total net amount by category")
            bar = (
                alt.Chart(cat_agg)
                .mark_bar()
                .encode(
                    x=alt.X("amount:Q", title="Net amount (dollars)"),
                    y=alt.Y("category:N", sort="-x", title="Category"),
                    tooltip=["category", "amount"],
                )
                .properties(height=400)
            )
            st.altair_chart(bar, use_container_width=True)

            st.markdown("### Category table")
            cat_agg["amount_fmt"] = cat_agg["amount"].apply(format_currency)
            st.dataframe(
                cat_agg[["category", "amount_fmt"]]
                .rename(columns={"amount_fmt": "amount"}),
                use_container_width=True,
                hide_index=True,
            )

    # Page: Forecast
    elif page == "Forecast":
        st.subheader("Monthly spend and simple forecast")

        if df_filtered.empty:
            st.warning("No data in selected range.")
        else:
            monthly_net = monthly_spend(df_filtered)

            st.markdown("### Historical monthly net spend")
            line = (
                alt.Chart(monthly_net)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("amount:Q", title="Net amount (dollars)"),
                    tooltip=["month:T", "amount:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(line, use_container_width=True)

            forecast_val = simple_forecast_next_month(monthly_net)

            if forecast_val is not None:
                st.info(
                    f"Forecast for next month: "
                    f"**{format_currency(forecast_val)}**"
                )
            else:
                st.info("Not enough data to compute forecast.")

            st.markdown("### Per-category forecast")
            monthly_cat = compute_monthly_totals(df_filtered)
            forecast_df = compute_forecast(monthly_cat)
            if forecast_df.empty:
                st.write("Not enough history to compute per-category forecasts yet.")
            else:
                forecast_df_display = forecast_df.copy()
                forecast_df_display["amount_forecast_fmt"] = forecast_df_display[
                    "amount_forecast"
                ].apply(format_currency)
                st.dataframe(
                    forecast_df_display[["year_month", "category", "amount_forecast_fmt"]]
                    .rename(columns={"amount_forecast_fmt": "amount_forecast"}),
                    use_container_width=True,
                    hide_index=True,
                )

    # Page: Subscriptions
    elif page == "Subscriptions":
        st.subheader("Subscriptions")

        subs_df = detect_subscriptions(df_filtered)
        if subs_df.empty:
            st.write("I don't see strong subscription patterns yet.")
        else:
            disp = subs_df.copy()
            disp["avg_amount_fmt"] = disp["avg_amount"].apply(format_currency)
            st.dataframe(
                disp[["merchant_raw", "months", "n_txns", "avg_amount_fmt", "std_amount"]],
                use_container_width=True,
                hide_index=True,
            )

    # Page: Unknown
    elif page == "Unknown":
        st.subheader("Transactions still labeled 'unknown'")

        # Ensure string dtype
        df_filtered["category"] = df_filtered["category"].astype("string")
        unknown_df = df_filtered[df_filtered["category"].str.lower().eq("unknown")].copy()

        if unknown_df.empty:
            st.write("No 'unknown' transactions right now")
        else:
            st.write(f"{len(unknown_df)} transactions are still 'unknown' (may need manual review).")
            st.dataframe(
                unknown_df.sort_values("date", ascending=False)
                          .loc[:, ["date", "merchant_raw", "amount", "account_id"]]
                          .assign(amount=lambda x: x["amount"].apply(format_currency))
                          .head(200),
                use_container_width=True,
                hide_index=True,
            )

    # Page: AI Assistant
    elif page == "AI Assistant":
        st.subheader("Ask the AI assistant about your finances")

        st.write(
            "Ask natural-language questions about your spending. "
            "On this screen, the assistant uses **all transactions in your dataset**, "
        )

        example = "Which category am I spending the most on, based on this data?"
        q = st.text_input(
            "Ask a question (example: 'How much did I spend on food in February?')",
            example,
        )
        if st.button("Ask", type="primary"):
            if df.empty:
                st.warning("There is no transaction data yet.")
            else:
                with st.spinner("Thinking..."):
                    # Use full df here, not df_filtered
                    answer = answer_question_llm(df, q)
                st.markdown("**Answer:**")
                st.write(answer)


if __name__ == "__main__":
    main()
