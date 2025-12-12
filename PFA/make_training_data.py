# make_training_data.py
from pathlib import Path
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)

# Where to write the training CSV
TRAIN_DIR = Path("data/training")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = TRAIN_DIR / "transactions_training.csv"

CATEGORIES = {
    "food": [
        "SAFEWAY", "TRADER JOE'S", "WALMART GROCERY", "COSTCO FOOD COURT",
        "STARBUCKS", "MCDONALD'S", "CHIPOTLE", "IN-N-OUT", "DOMINO'S PIZZA",
        "PIZZA HUT", "UBER EATS", "DOORDASH", "PANDA EXPRESS", "TACO BELL",
    ],
    "rent": [
        "RENT PAYMENT", "PROPERTY MANAGEMENT CO", "APARTMENT RENT",
    ],
    "utilities": [
        "PG&E", "COMCAST INTERNET", "T-MOBILE", "AT&T INTERNET",
        "CITY WATER & SEWER", "SMUD UTILITIES",
    ],
    "subscriptions": [
        "NETFLIX", "SPOTIFY", "APPLE ICLOUD", "DISNEY+", "HULU",
        "YOUTUBE PREMIUM", "MIDJOURNEY SUBSCRIPTION",
    ],
    "shopping": [
        "AMAZON MARKETPLACE", "TARGET", "WALMART SUPERSTORE",
        "BEST BUY", "NIKE FACTORY STORE", "APPLE STORE", "UNI QLO",
        "COSTCO WHOLESALE", "HOME DEPOT", "LOWE'S",
    ],
    "transportation": [
        "SHELL GAS", "CHEVRON", "ARCO", "EXXONMOBIL", "UBER", "LYFT",
        "BART CLIPPER", "METRO TRANSIT",
    ],
    "entertainment": [
        "AMC THEATERS", "REGAL CINEMAS", "STEAM PURCHASE",
        "NINTENDO ESHOP", "PLAYSTATION STORE", "CONCERT TICKETS",
        "BOWLING ALLEY", "ESCAPE ROOM",
    ],
    "health": [
        "WALGREENS PHARMACY", "CVS PHARMACY", "DENTAL CLINIC",
        "OPTOMETRY CENTER", "URGENT CARE", "GYM MEMBERSHIP",
        "24 HOUR FITNESS", "PLANET FITNESS",
    ],
    "other": [
        "VENMO PAYMENT", "PAYPAL *EBAY", "ZELLE PAYMENT",
        "BANK FEE", "CHECK DEPOSIT", "CASH APP",
    ],
    "income": [
        "DIRECT DEPOSIT PAYROLL",
        "FREELANCE PAYMENT",
        "REFUND AMAZON",
        "TAX REFUND",
        "CASH DEPOSIT",
        "VENMO INCOMING",
        "ZELLE INCOMING",
    ],
}

# rough target counts per category (must sum to 500)
CATEGORY_COUNTS = {
    "food": 100,
    "rent": 40,
    "utilities": 60,
    "subscriptions": 60,
    "shopping": 70,
    "transportation": 50,
    "entertainment": 40,
    "health": 30,
    "other": 20,
    "income": 30,
}
assert sum(CATEGORY_COUNTS.values()) == 500, "Counts must sum to 500"


def random_date(start: date, end: date) -> date:
    #Pick a random date between start and end
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def sample_amount(category: str) -> float:
    
    # Sample a realistic amount based on category
    # -Expenses are negative
    # -Income is positive
    if category == "income":
        # mix of paychecks and smaller incoming amounts
        base = random.choice([500, 800, 1200, 1800, 2200, 2600, 3200])
        noise = random.uniform(-100, 150)
        return round(base + noise, 2)

    if category == "rent":
        base = random.choice([1650, 1750, 1800, 1850, 1900])
        noise = random.uniform(-50, 50)
        return -(base + noise)

    if category == "food":
        base = random.uniform(6, 60)
        return -round(base, 2)

    if category == "utilities":
        base = random.uniform(40, 140)
        return -round(base, 2)

    if category == "subscriptions":
        base = random.choice([4.99, 7.99, 9.99, 12.99, 14.99, 19.99])
        noise = random.uniform(-1, 1)
        return -round(base + noise, 2)

    if category == "shopping":
        base = random.uniform(15, 250)
        return -round(base, 2)

    if category == "transportation":
        base = random.uniform(8, 80)
        return -round(base, 2)

    if category == "entertainment":
        base = random.uniform(10, 150)
        return -round(base, 2)

    if category == "health":
        base = random.uniform(10, 250)
        return -round(base, 2)

    if category == "other":
        base = random.uniform(5, 120)
        return -round(base, 2)

    # fallback
    return -round(random.uniform(5, 100), 2)


def main():
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)

    rows = []

    for category, count in CATEGORY_COUNTS.items():
        merchants = CATEGORIES[category]
        for _ in range(count):
            d = random_date(start_date, end_date)
            merchant = random.choice(merchants)
            amount = sample_amount(category)

            # can use multiple account_ids
            account_id = random.choice(["checking", "credit_card", "savings"])

            rows.append(
                {
                    "date": d.isoformat(),
                    "account_id": account_id,
                    "merchant_raw": merchant,
                    "amount": amount,
                    "category": category,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
