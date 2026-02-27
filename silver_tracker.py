"""
TATAGOLD ETF Tracker
- Tracks TATAGOLD.NS (Tata Gold ETF) via Yahoo Finance
- Calculates iNAV using international gold spot price + USD/INR + IMPORT DUTY

iNAV Formula (3 steps):
  1. intl_inr_gram = (Gold_USD_oz x USD_INR) / 31.1035
  2. dom_inr_gram  = intl_inr_gram x (1 + 0.097)     ← Indian import duty + GST
  3. iNAV          = dom_inr_gram  x GOLD_GRAMS_PER_UNIT (0.001728g)

Import duty breakdown (post Jul-2024 Union Budget):
  5.0% basic customs + 1.0% AIDC + 0.5% SWS + 3.0% GST = 9.7% total

GOLD_GRAMS_PER_UNIT = 0.001728g
  Calibrated from Tata AMC NAV on 26-Feb-2026 = Rs 15.252
  (Gold $2,900/oz, USD/INR 86.30 → iNAV = Rs 15.252 ✅)
  Get exact value from Tata AMC SID/factsheet to confirm.

- Reports premium/discount, buy/sell suggestion, dollar rate
- Sends Telegram alert and updates CSV on every run
"""

import yfinance as yf
import pandas as pd
import requests
import os
import sys
import traceback
from datetime import datetime
import pytz

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TROY_OZ_TO_GRAM     = 31.1035    # grams in 1 troy ounce

# TATAGOLD: ~0.001728g/unit (calibrated from Tata AMC NAV 26-Feb-2026)
# Get the exact value from the Tata Gold ETF SID document to confirm.
GOLD_GRAMS_PER_UNIT = 0.001728

# Indian gold import duty effective rate (post Jul-2024 budget)
# = 5% basic customs + 1% AIDC + 0.5% SWS + 3% GST = ~9.7% total
GOLD_IMPORT_DUTY    = 0.097

EXPENSE_RATIO       = 0.0038     # 0.38% annual TER (as of Feb 2026)
CSV_FILE            = "tatagold_log.csv"
IST                 = pytz.timezone("Asia/Kolkata")

# Buy/Sell thresholds (premium/discount %)
SELL_THRESHOLD      =  1.0   # sell if premium > 1%
BUY_THRESHOLD       = -1.0   # buy  if discount > 1%


# ─────────────────────────────────────────────
# PRICE FETCHING
# ─────────────────────────────────────────────
def fetch_price(ticker: str, label: str) -> float:
    """Fetch latest close price for a Yahoo Finance ticker. Raises on failure."""
    t = yf.Ticker(ticker)
    for period, interval in [("1d", "1m"), ("5d", "1d")]:
        hist = t.history(period=period, interval=interval)
        if not hist.empty:
            price = float(hist["Close"].dropna().iloc[-1])
            print(f"  [{label}] {ticker} -> {price:.4f}")
            return price
    raise ValueError(f"No data for {ticker}")


def get_all_prices():
    print("Fetching prices from Yahoo Finance...")

    # Gold spot (USD per troy oz)
    gold_usd = None
    for ticker in ["GC=F", "XAUUSD=X"]:
        try:
            gold_usd = fetch_price(ticker, "Gold USD/oz")
            break
        except Exception:
            continue
    if gold_usd is None:
        raise ValueError("Could not fetch gold spot price from GC=F or XAUUSD=X")

    # USD/INR rate
    usd_inr   = fetch_price("USDINR=X", "USD/INR")

    # TATAGOLD ETF price (INR)
    etf_price = fetch_price("TATAGOLD.NS", "TATAGOLD.NS")

    return gold_usd, usd_inr, etf_price


# ─────────────────────────────────────────────
# iNAV CALCULATION
# ─────────────────────────────────────────────
def calculate_inav(gold_usd: float, usd_inr: float) -> tuple[float, float, float]:
    """
    3-step iNAV calculation:
      1. Convert international gold price to INR per gram
      2. Apply Indian import duty + GST (~9.7%) to get domestic price
      3. Multiply by grams per unit
    Returns: (iNAV, domestic_inr_per_gram, international_inr_per_gram)
    """
    intl_inr_gram = (gold_usd * usd_inr) / TROY_OZ_TO_GRAM
    dom_inr_gram  = intl_inr_gram * (1 + GOLD_IMPORT_DUTY)
    inav          = dom_inr_gram * GOLD_GRAMS_PER_UNIT
    return round(inav, 4), round(dom_inr_gram, 4), round(intl_inr_gram, 4)


def calculate_premium_discount(etf_price: float, inav: float) -> float:
    return round(((etf_price - inav) / inav) * 100, 3)


def get_suggestion(premium_pct: float) -> str:
    if premium_pct > SELL_THRESHOLD:
        return f"SELL  (Premium +{premium_pct:.2f}% > {SELL_THRESHOLD}%)"
    elif premium_pct < BUY_THRESHOLD:
        return f"BUY   (Discount {premium_pct:.2f}% < {BUY_THRESHOLD}%)"
    else:
        return f"HOLD  (Near Fair Value: {premium_pct:+.2f}%)"


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Warning: Telegram credentials missing - skipping notification.")
        return
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print(f"Telegram sent OK (status {resp.status_code})")
    except requests.exceptions.HTTPError as e:
        print(f"Telegram HTTP error: {e} | Response: {resp.text}")
    except Exception as e:
        print(f"Telegram error: {e}")


def build_telegram_message(data: dict) -> str:
    pct   = data["premium_discount_pct"]
    arrow = "UP" if pct > 0 else "DN"
    return (
        f"TATAGOLD Tracker\n"
        f"Time       : {data['timestamp']}\n"
        f"ETF Price  : Rs {data['etf_price_inr']}\n"
        f"iNAV       : Rs {data['inav_inr']}\n"
        f"Prem/Disc  : {pct:+.2f}% [{arrow}]\n"
        f"USD/INR    : Rs {data['usd_inr']}\n"
        f"Gold (Intl): ${data['gold_usd_oz']}/oz\n"
        f"Gold (Dom) : Rs {data['gold_dom_inr_gram']}/g  (incl. 9.7% duty+GST)\n"
        f"Signal     : {data['suggestion']}"
    )


# ─────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────
def update_csv(data: dict):
    df_new = pd.DataFrame([data])
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            df_existing = pd.read_csv(CSV_FILE)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing CSV ({e}), starting fresh.")
            df = df_new
    else:
        df = df_new
    df.to_csv(CSV_FILE, index=False)
    print(f"CSV updated -> {CSV_FILE}  (rows: {len(df)})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"\n{'='*55}")
    print(f"  TATAGOLD Tracker  |  {now}")
    print(f"{'='*55}")

    try:
        gold_usd, usd_inr, etf_price         = get_all_prices()
        inav, dom_inr_gram, intl_inr_gram     = calculate_inav(gold_usd, usd_inr)
        premium_pct                           = calculate_premium_discount(etf_price, inav)
        suggestion                            = get_suggestion(premium_pct)

        data = {
            "timestamp"            : now,
            "etf_price_inr"        : round(etf_price,      2),
            "inav_inr"             : round(inav,            4),
            "premium_discount_pct" : round(premium_pct,     3),
            "gold_usd_oz"          : round(gold_usd,        4),
            "usd_inr"              : round(usd_inr,         4),
            "gold_intl_inr_gram"   : round(intl_inr_gram,   2),
            "gold_dom_inr_gram"    : round(dom_inr_gram,    2),
            "suggestion"           : suggestion,
        }

        print(f"\n{'─'*45}")
        for k, v in data.items():
            print(f"  {k:<25}: {v}")
        print(f"{'─'*45}\n")

        update_csv(data)
        send_telegram(build_telegram_message(data))

        print("Run complete.\n")

    except Exception as e:
        err_msg = f"TATAGOLD Tracker ERROR\n{now}\n\n{str(e)}"
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        send_telegram(err_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()