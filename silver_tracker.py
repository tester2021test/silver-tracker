import yfinance as yf
import pandas as pd
import pandas_ta as ta
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
def send_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "Markdown"
    })

# ===================== HELPERS =====================
def get_price(symbol):
    df = yf.Ticker(symbol).history(period="5d")
    if df.empty:
        return 0.0, 0.0
    if len(df) == 1:
        return df["Close"].iloc[-1], df["Close"].iloc[-1]
    return df["Close"].iloc[-1], df["Close"].iloc[-2]

def save_csv(row):
    file = Path("silver_history.csv")
    write_header = not file.exists()
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ===================== MAIN =====================
def main():
    tickers = {
        "Silver": "SI=F",
        "USDINR": "INR=X",
        "ETF": "TATSILV.NS",
        "VIX": "^INDIAVIX"
    }

    live, prev = {}, {}
    for k, t in tickers.items():
        live[k], prev[k] = get_price(t)

    etf_intra = yf.download("TATSILV.NS", period="5d", interval="15m", progress=False)["Close"].dropna()

    rsi = 50
    if len(etf_intra) > 14:
        rsi_val = ta.rsi(etf_intra, length=14)
        if rsi_val is not None and not rsi_val.dropna().empty:
            rsi = float(rsi_val.dropna().iloc[-1])

    silver_prev = prev["Silver"] * prev["USDINR"]
    silver_curr = live["Silver"] * live["USDINR"]
    silver_move = (silver_curr - silver_prev) / silver_prev if silver_prev else 0

    fair_inav = prev["ETF"] * (1 + silver_move)
    premium = (live["ETF"] - fair_inav) / fair_inav * 100 if fair_inav else 0

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t = now.time()

    phase = "LIVE"
    if t < time(9, 15):
        phase = "PRE-MARKET"
    elif t > time(15, 30):
        phase = "POST-MARKET"

    signal = "NEUTRAL"
    if premium < -2:
        signal = "STRONG BUY"
    elif premium > 2:
        signal = "AVOID"

    save_csv({
        "timestamp": now.isoformat(),
        "phase": phase,
        "etf_price": round(live["ETF"], 2),
        "fair_inav": round(fair_inav, 2),
        "premium_pct": round(premium, 2),
        "rsi": round(rsi, 1),
        "signal": signal
    })

    if abs(premium) >= 2:
        send_telegram(
            f"🚨 *Silver ETF Alert*\n\n"
            f"Phase: *{phase}*\n"
            f"ETF: ₹{live['ETF']:.2f}\n"
            f"Fair: ₹{fair_inav:.2f}\n"
            f"Premium: *{premium:+.2f}%*\n"
            f"RSI: {rsi:.1f}\n"
            f"Signal: *{signal}*"
        )

    print("Run completed", now)

if __name__ == "__main__":
    main()
