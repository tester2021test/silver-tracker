"""
TATSILVER ETF Tracker
- Tracks TATSILV.NS (Tata Silver ETF) via Yahoo Finance
- Calculates iNAV using international silver spot price + USD/INR
- iNAV formula: (Silver_USD_per_troy_oz × USD_INR) / 31.1035 grams_per_oz × 1 gram_per_unit
- Reports premium/discount, buy/sell suggestion, dollar rate
- Sends Telegram alert and updates GitHub CSV on every run
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
TROY_OZ_TO_GRAM     = 31.1035     # grams in 1 troy ounce
SILVER_GRAMS_PER_UNIT = 1.0       # TATSILVER: 1 unit ≈ 1 gram silver
EXPENSE_RATIO       = 0.0044      # 0.44% annual; used to adjust iNAV if desired
CSV_FILE            = "tatsilver_log.csv"
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
    # Try 1-min intraday first, fallback to daily
    for period, interval in [("1d", "1m"), ("5d", "1d")]:
        hist = t.history(period=period, interval=interval)
        if not hist.empty:
            price = float(hist["Close"].dropna().iloc[-1])
            print(f"  [{label}] {ticker} → {price:.4f}")
            return price
    raise ValueError(f"No data for {ticker}")


def get_all_prices():
    print("📡 Fetching prices from Yahoo Finance...")

    # Silver spot (USD per troy oz)
    silver_usd = None
    for ticker in ["SI=F", "XAGUSD=X"]:
        try:
            silver_usd = fetch_price(ticker, "Silver USD/oz")
            break
        except Exception:
            continue
    if silver_usd is None:
        raise ValueError("Could not fetch silver spot price from SI=F or XAGUSD=X")

    # USD/INR rate
    usd_inr = fetch_price("USDINR=X", "USD/INR")

    # TATSILVER ETF price (INR)
    etf_price = fetch_price("TATSILV.NS", "TATSILV.NS")

    return silver_usd, usd_inr, etf_price


# ─────────────────────────────────────────────
# iNAV CALCULATION
# ─────────────────────────────────────────────
def calculate_inav(silver_usd: float, usd_inr: float) -> tuple[float, float]:
    """
    iNAV (₹ per unit) = silver price in INR per gram × grams per unit
    Silver in INR/gram = (Silver_USD/troy_oz × USD/INR) / 31.1035
    """
    silver_inr_per_gram = (silver_usd * usd_inr) / TROY_OZ_TO_GRAM
    inav = silver_inr_per_gram * SILVER_GRAMS_PER_UNIT
    return round(inav, 4), round(silver_inr_per_gram, 4)


def calculate_premium_discount(etf_price: float, inav: float) -> float:
    return round(((etf_price - inav) / inav) * 100, 3)


def get_suggestion(premium_pct: float) -> str:
    if premium_pct > SELL_THRESHOLD:
        return f"🔴 SELL  (Premium +{premium_pct:.2f}% > {SELL_THRESHOLD}%)"
    elif premium_pct < BUY_THRESHOLD:
        return f"🟢 BUY   (Discount {premium_pct:.2f}% < {BUY_THRESHOLD}%)"
    else:
        return f"🟡 HOLD  (Near Fair Value: {premium_pct:+.2f}%)"


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("⚠️  Telegram credentials missing — skipping notification.")
        return
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print(f"✅ Telegram sent (status {resp.status_code})")
    except Exception as e:
        print(f"❌ Telegram error: {e}")


def build_telegram_message(data: dict) -> str:
    pct   = data["premium_discount_pct"]
    arrow = "🔺" if pct > 0 else "🔻"
    return (
        f"<b>🥈 TATSILVER Tracker</b>\n"
        f"⏰ {data['timestamp']}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 ETF Price : ₹{data['etf_price_inr']}\n"
        f"📊 iNAV      : ₹{data['inav_inr']}\n"
        f"{arrow} Prem/Disc  : {pct:+.2f}%\n"
        f"💵 USD/INR   : ₹{data['usd_inr']}\n"
        f"🪙 Silver    : ${data['silver_usd_oz']}/oz\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🎯 {data['suggestion']}"
    )


# ─────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────
def update_csv(data: dict):
    df_new = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_FILE, index=False)
    print(f"📁 CSV updated → {CSV_FILE}  (rows: {len(df)})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"\n{'='*50}")
    print(f"  TATSILVER Tracker  |  {now}")
    print(f"{'='*50}")

    try:
        silver_usd, usd_inr, etf_price = get_all_prices()
        inav, silver_inr_gram          = calculate_inav(silver_usd, usd_inr)
        premium_pct                    = calculate_premium_discount(etf_price, inav)
        suggestion                     = get_suggestion(premium_pct)

        data = {
            "timestamp"           : now,
            "etf_price_inr"       : round(etf_price,      2),
            "inav_inr"            : round(inav,            2),
            "premium_discount_pct": round(premium_pct,     3),
            "silver_usd_oz"       : round(silver_usd,      4),
            "usd_inr"             : round(usd_inr,         4),
            "silver_inr_gram"     : round(silver_inr_gram, 4),
            "suggestion"          : suggestion,
        }

        # Print summary
        print(f"\n{'─'*40}")
        for k, v in data.items():
            print(f"  {k:<25}: {v}")
        print(f"{'─'*40}\n")

        # Update CSV
        update_csv(data)

        # Send Telegram
        msg = build_telegram_message(data)
        send_telegram(msg)

        print("✅ Run complete.\n")

    except Exception as e:
        err_msg = f"⚠️ <b>TATSILVER Tracker ERROR</b>\n{now}\n\n<code>{str(e)}</code>"
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()
        send_telegram(err_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
