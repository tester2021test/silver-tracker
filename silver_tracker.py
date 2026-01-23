# ============================================================
# 💎 Ultimate Silver Tracker (GitHub Actions – Production Safe)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
import os
import csv
from pathlib import Path
from datetime import datetime, time

# ===================== CONFIG =====================
KG_CONVERSION = 32.1507466
IMPORT_DUTY = 0.06
GST_RATE = 0.03

# ===================== TELEGRAM =====================
def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        },
        timeout=10
    )

# ===================== SAFE PRICE FETCH =====================
def get_live_prev(symbol):
    df = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
    if df.empty:
        return 0.0, 0.0
    if len(df) == 1:
        close = float(df["Close"].iloc[-1])
        return close, close
    return float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])

# ===================== RSI (MANUAL & STABLE) =====================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===================== CSV STORAGE =====================
def save_csv(row):
    file = Path("silver_history.csv")
    write_header = not file.exists()

    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ===================== MAIN ENGINE =====================
def main():
    print("⏳ Running Silver Tracker...")

    tickers = {
        "Silver_Global": "SI=F",
        "USD_INR": "INR=X",
        "ETF": "TATSILV.NS",
        "India_VIX": "^INDIAVIX"
    }

    live, prev = {}, {}
    for k, sym in tickers.items():
        live[k], prev[k] = get_live_prev(sym)

    # ---------- Intraday data for RSI ----------
    etf_intra = yf.download(
        "TATSILV.NS",
        period="5d",
        interval="15m",
        auto_adjust=True,
        progress=False
    )["Close"].dropna()

    rsi = 50.0
    if len(etf_intra) > 14:
        rsi_series = calculate_rsi(etf_intra, 14)
        if not rsi_series.dropna().empty:
            rsi = float(rsi_series.dropna().iloc[-1])

    # ---------- Silver move ----------
    prev_inr = prev["Silver_Global"] * prev["USD_INR"]
    curr_inr = live["Silver_Global"] * live["USD_INR"]

    silver_move_pct = (
        (curr_inr - prev_inr) / prev_inr
        if prev_inr > 0 else 0.0
    )

    # ---------- ETF Fair iNAV ----------
    fair_inav = prev["ETF"] * (1 + silver_move_pct) if prev["ETF"] > 0 else 0.0
    premium_pct = (
        (live["ETF"] - fair_inav) / fair_inav * 100
        if fair_inav > 0 else 0.0
    )

    # ---------- Market Phase ----------
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    current_time = now.time()

    if current_time < time(9, 15):
        market_phase = "PRE-MARKET"
    elif current_time > time(15, 30):
        market_phase = "POST-MARKET"
    else:
        market_phase = "LIVE"

    # ---------- Signal ----------
    signal = "NEUTRAL"
    if premium_pct <= -2:
        signal = "STRONG BUY"
    elif premium_pct >= 2:
        signal = "AVOID / PROFIT BOOK"

    # ---------- Save History ----------
    save_csv({
        "timestamp": now.isoformat(),
        "market_phase": market_phase,
        "etf_price": round(live["ETF"], 2),
        "fair_inav": round(fair_inav, 2),
        "premium_pct": round(premium_pct, 2),
        "rsi": round(rsi, 1),
        "signal": signal
    })

    # ---------- Telegram Alert (ONLY if |premium| ≥ 2%) ----------
    if abs(premium_pct) >= 2:
        send_telegram(
            f"🚨 *Silver ETF Alert*\n\n"
            f"🕒 Phase: *{market_phase}*\n"
            f"ETF Price: ₹{live['ETF']:.2f}\n"
            f"Fair iNAV: ₹{fair_inav:.2f}\n"
            f"Premium: *{premium_pct:+.2f}%*\n"
            f"RSI (15m): {rsi:.1f}\n"
            f"Signal: *{signal}*"
        )

    print(f"✅ Completed | {now.strftime('%Y-%m-%d %H:%M:%S IST')}")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
