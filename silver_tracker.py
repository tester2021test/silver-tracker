"""
TATAGOLD ETF Tracker
- Tracks TATAGOLD.NS (Tata Gold ETF) via Yahoo Finance
- Calculates iNAV using international gold spot price + USD/INR + Import Duty

iNAV Formula (3 steps):
  1. intl_inr_gram = (Gold_USD_oz x USD_INR) / 31.1035
  2. dom_inr_gram  = intl_inr_gram x (1 + 0.065)   <- 6.5% import duty
  3. iNAV          = dom_inr_gram  x 0.000955g

Import duty breakdown (post Jul-2024 budget, for ETF fund - NO GST):
  5.0% basic customs + 1.0% AIDC + 0.5% SWS = 6.5% total
  (GST of 3% is NOT applied here — ETF funds hold gold as investment, not retail sale)

Calibration (27-Feb-2026, verified against Tata AMC & Groww):
  Gold = $5,177/oz, USD/INR = 91, Domestic 24K = Rs 16,023/gram
  Tata AMC iNAV = Rs 15.34  |  Our calc = Rs 15.41  (0.46% error — timing gap only)
  GOLD_GRAMS_PER_UNIT = 0.000955g
    -> Method A: Tata iNAV 15.34 / Groww Rs 16,023/gram   = 0.000957g
    -> Method B: Tata NAV  15.252 / 26-Feb Rs 15,970/gram  = 0.000955g
    -> Average: 0.000956g  -> using 0.000955g

Previous errors that have been corrected:
  - GPU was 0.001728g (WRONG: calibrated assuming gold was ~$2,900 and USD/INR ~86)
  - DUTY was 9.7%     (WRONG: 9.7% included 3% GST; ETF funds don't pay GST on holdings)
  - Bounds were $2000-$4000 (OUTDATED: gold is $5,177 as of Feb 2026)
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
TROY_OZ_TO_GRAM     = 31.1035

# Calibrated 27-Feb-2026 from Tata AMC iNAV + Groww domestic gold price
# Gold $5177/oz, USD/INR 91 -> domestic Rs 16,023/gram -> GPU = 15.34 / 16023 = 0.000957g
GOLD_GRAMS_PER_UNIT = 0.000955

# Indian gold import duty for ETF funds (post Jul-2024 budget, NO GST)
# 5% basic customs + 1% AIDC + 0.5% SWS = 6.5%
GOLD_IMPORT_DUTY    = 0.065

CSV_FILE            = "tatagold_log.csv"
IST                 = pytz.timezone("Asia/Kolkata")

SELL_THRESHOLD      =  1.0
BUY_THRESHOLD       = -1.0

# ─── Sanity bounds — reject obviously wrong data before using ─
BOUNDS = {
    "gold_usd" : (4000.0, 8000.0),   # COMEX gold USD/troy oz (as of Feb 2026: ~$5,177)
    "usd_inr"  : (82.0,  105.0),     # USD/INR (as of Feb 2026: ~91)
    "etf"      : (10.0,   60.0),     # TATAGOLD.NS price in INR
}


# ─────────────────────────────────────────────
# PRICE FETCHING  (with validation)
# ─────────────────────────────────────────────
def fetch_and_validate(ticker: str, label: str,
                       lo: float, hi: float) -> float:
    """
    Fetch latest price. Tries daily close first (more stable),
    then intraday. Validates against expected range.
    """
    periods = [("5d", "1d"), ("1d", "1m"), ("1mo", "1d")]
    t       = yf.Ticker(ticker)
    last_err = None

    for period, interval in periods:
        try:
            hist = t.history(period=period, interval=interval)
            if hist.empty:
                last_err = f"empty ({period}/{interval})"
                continue

            price = float(hist["Close"].dropna().iloc[-1])

            if not (lo <= price <= hi):
                print(f"  [{label}] {ticker} -> {price:.4f}  *** OUT OF RANGE [{lo}, {hi}] — skipping ***")
                last_err = f"price {price:.4f} out of range [{lo}, {hi}]"
                continue

            print(f"  [{label}] {ticker} -> {price:.4f}  [OK]")
            return price

        except Exception as e:
            last_err = str(e)
            continue

    raise ValueError(
        f"Could not get valid price for {ticker} ({label}). "
        f"Expected [{lo}, {hi}]. Last issue: {last_err}"
    )


def get_all_prices():
    print("Fetching prices from Yahoo Finance...")

    # ── Gold spot: GC=F first, fallback to XAUUSD=X ──────────
    lo_g, hi_g   = BOUNDS["gold_usd"]
    lo_fx, hi_fx = BOUNDS["usd_inr"]
    lo_e, hi_e   = BOUNDS["etf"]

    gold_usd = None
    for ticker in ["GC=F", "XAUUSD=X"]:
        try:
            gold_usd = fetch_and_validate(ticker, "Gold USD/oz", lo_g, hi_g)
            break
        except ValueError as e:
            print(f"  [Gold] {ticker} failed: {e}")

    if gold_usd is None:
        raise ValueError(
            f"Could not fetch valid gold price from GC=F or XAUUSD=X. "
            f"Expected ${lo_g}–${hi_g}/oz. Current gold price is ~$5,177/oz (Feb 2026). "
            f"Update BOUNDS['gold_usd'] if gold has moved significantly."
        )

    usd_inr   = fetch_and_validate("USDINR=X",    "USD/INR",     lo_fx, hi_fx)
    etf_price = fetch_and_validate("TATAGOLD.NS", "TATAGOLD.NS", lo_e,  hi_e)

    return gold_usd, usd_inr, etf_price


# ─────────────────────────────────────────────
# iNAV CALCULATION
# ─────────────────────────────────────────────
def calculate_inav(gold_usd: float, usd_inr: float) -> tuple[float, float, float]:
    """
    Step 1: COMEX gold -> INR/gram (international)
    Step 2: Apply 6.5% import duty -> domestic INR/gram
    Step 3: Multiply by 0.000955g per unit
    Returns: (iNAV, domestic_inr_per_gram, international_inr_per_gram)
    """
    intl_inr_gram = (gold_usd * usd_inr) / TROY_OZ_TO_GRAM
    dom_inr_gram  = intl_inr_gram * (1 + GOLD_IMPORT_DUTY)
    inav          = dom_inr_gram * GOLD_GRAMS_PER_UNIT
    return round(inav, 4), round(dom_inr_gram, 2), round(intl_inr_gram, 2)


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
        f"Signal     : {data['suggestion']}\n"
        f"---\n"
        f"Gold (COMEX): ${data['gold_usd_oz']}/oz\n"
        f"USD/INR     : Rs {data['usd_inr']}\n"
        f"Gold (Intl) : Rs {data['gold_intl_inr_gram']}/g\n"
        f"Gold (Dom)  : Rs {data['gold_dom_inr_gram']}/g  (+6.5% duty)"
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
        gold_usd, usd_inr, etf_price      = get_all_prices()
        inav, dom_inr_gram, intl_inr_gram  = calculate_inav(gold_usd, usd_inr)
        premium_pct                        = calculate_premium_discount(etf_price, inav)
        suggestion                         = get_suggestion(premium_pct)

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