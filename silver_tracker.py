# ============================================================
# 💎 Ultimate Silver Tracker — Enhanced Edition
# Features:
#   ✅ Multi-asset: Silver ETF, Gold ETF, MCX Silver proxy
#   ✅ Indicators: RSI, MACD, Bollinger Bands, ATR, VWAP
#   ✅ Smarter signals: Multi-factor confirmation + risk scoring
#   ✅ Retry logic & error handling on all fetches
#   ✅ Historical backtesting (last 30 days)
#   ✅ Rich Telegram alerts (formatted tables, emoji status)
#   ✅ CSV history with extended columns
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
import os
import csv
import time as time_module
import json
from pathlib import Path
from datetime import datetime, time, timedelta
from functools import wraps

# ===================== CONFIG =====================
KG_CONVERSION   = 32.1507466
IMPORT_DUTY     = 0.06
GST_RATE        = 0.03
PREMIUM_BUY     = -2.0      # % discount → BUY signal
PREMIUM_SELL    = 2.0       # % premium → SELL signal
RSI_OVERSOLD    = 40
RSI_OVERBOUGHT  = 65
ATR_RISK_MULT   = 1.5       # Stop = entry − ATR × multiplier
MAX_RETRIES     = 3
RETRY_DELAY     = 4         # seconds between retries

TICKERS = {
    "Silver_Global" : "SI=F",
    "Gold_Global"   : "GC=F",
    "USD_INR"       : "INR=X",
    "Silver_ETF"    : "TATSILV.NS",
    "Gold_ETF"      : "GOLDBEES.NS",
    "MCX_Silver"    : "SILVERM.MCX",   # may not always trade; handled gracefully
    "India_VIX"     : "^INDIAVIX",
}

# ===================== RETRY DECORATOR =====================
def with_retry(retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    last_err = e
                    print(f"⚠️  Attempt {attempt}/{retries} failed for {fn.__name__}: {e}")
                    if attempt < retries:
                        time_module.sleep(delay)
            print(f"❌ All {retries} retries failed for {fn.__name__}: {last_err}")
            return None
        return wrapper
    return decorator

# ===================== FETCH =====================
@with_retry()
def fetch_history(symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty dataframe for {symbol}")
    return df

def safe_close(df: pd.DataFrame) -> pd.Series:
    """Return 1-D Close series from potentially multi-index df."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)

def get_live_prev(symbol: str):
    """Returns (live_price, prev_close). Falls back to (0, 0) on failure."""
    df = fetch_history(symbol, period="5d", interval="1d")
    if df is None or df.empty:
        return 0.0, 0.0
    close = safe_close(df)
    if len(close) == 1:
        return float(close.iloc[-1]), float(close.iloc[-1])
    return float(close.iloc[-1]), float(close.iloc[-2])

# ===================== INDICATORS =====================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta   = series.diff()
    gain    = delta.clip(lower=0)
    loss    = -delta.clip(upper=0)
    avg_g   = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l   = loss.ewm(com=period - 1, min_periods=period).mean()
    rs      = avg_g / avg_l.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram) as floats (last value)."""
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return (
        round(float(macd_line.iloc[-1]),  4),
        round(float(sig_line.iloc[-1]),   4),
        round(float(histogram.iloc[-1]),  4),
    )

def calculate_bollinger(series: pd.Series, period=20, std_dev=2):
    """Returns (upper, mid, lower, %B) as floats (last value)."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return (
        round(float(upper.iloc[-1]),  4),
        round(float(mid.iloc[-1]),    4),
        round(float(lower.iloc[-1]),  4),
        round(float(pct_b.iloc[-1]),  4),
    )

def calculate_atr(df: pd.DataFrame, period=14) -> float:
    """Average True Range — proxy for volatility / stop distance."""
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = safe_close(df)
    prev  = close.shift(1)
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 4) if not atr.empty else 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    """Intraday VWAP approximation using OHLC/4 × Volume."""
    close  = safe_close(df)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    typical = (close + high + low) / 3
    vwap   = (typical * volume).cumsum() / volume.cumsum()
    return round(float(vwap.iloc[-1]), 4)

# ===================== SIGNAL ENGINE =====================
def build_signal(premium_pct: float, rsi: float, macd_hist: float,
                 pct_b: float, vix: float) -> dict:
    """
    Multi-factor signal scoring.
    Each factor contributes a score in [-2, +2].
    Total score → action + confidence.

    Positive score  = BULLISH  (BUY)
    Negative score  = BEARISH  (AVOID/SELL)
    """
    score = 0
    reasons = []

    # --- Premium / Discount ---
    if premium_pct <= PREMIUM_BUY:
        score += 2
        reasons.append(f"ETF discount {premium_pct:+.1f}%")
    elif premium_pct >= PREMIUM_SELL:
        score -= 2
        reasons.append(f"ETF premium {premium_pct:+.1f}%")
    else:
        reasons.append(f"Premium neutral {premium_pct:+.1f}%")

    # --- RSI ---
    if rsi < RSI_OVERSOLD:
        score += 2
        reasons.append(f"RSI oversold {rsi:.1f}")
    elif rsi > RSI_OVERBOUGHT:
        score -= 1
        reasons.append(f"RSI elevated {rsi:.1f}")
    else:
        reasons.append(f"RSI neutral {rsi:.1f}")

    # --- MACD histogram direction ---
    if macd_hist > 0:
        score += 1
        reasons.append("MACD bullish crossover")
    elif macd_hist < 0:
        score -= 1
        reasons.append("MACD bearish crossover")

    # --- Bollinger %B ---
    if pct_b < 0.2:
        score += 1
        reasons.append(f"BB lower band touch ({pct_b:.2f})")
    elif pct_b > 0.8:
        score -= 1
        reasons.append(f"BB upper band touch ({pct_b:.2f})")

    # --- VIX risk filter ---
    if vix > 20:
        score -= 1
        reasons.append(f"High VIX {vix:.1f} → risk-off")

    # Translate score
    if score >= 4:
        action, emoji = "STRONG BUY",      "🟢🟢"
    elif score >= 2:
        action, emoji = "BUY",             "🟢"
    elif score <= -4:
        action, emoji = "STRONG SELL",     "🔴🔴"
    elif score <= -2:
        action, emoji = "AVOID / SELL",    "🔴"
    else:
        action, emoji = "NEUTRAL / HOLD",  "🟡"

    confidence = min(abs(score) / 7 * 100, 100)

    return {
        "action":     action,
        "emoji":      emoji,
        "score":      score,
        "confidence": round(confidence, 1),
        "reasons":    reasons,
    }

# ===================== RISK MANAGEMENT =====================
def risk_management(etf_price: float, atr: float, signal_score: int) -> dict:
    """Returns suggested stop-loss, target, and position-size note."""
    stop_loss   = round(etf_price - ATR_RISK_MULT * atr, 2)
    target_1r   = round(etf_price + ATR_RISK_MULT * atr, 2)       # 1:1 RR
    target_2r   = round(etf_price + 2 * ATR_RISK_MULT * atr, 2)   # 1:2 RR
    risk_pct    = round((etf_price - stop_loss) / etf_price * 100, 2)

    if abs(signal_score) >= 4:
        size_note = "Full position (high conviction)"
    elif abs(signal_score) >= 2:
        size_note = "Half position (moderate conviction)"
    else:
        size_note = "No position / wait for confirmation"

    return {
        "stop_loss" : stop_loss,
        "target_1r" : target_1r,
        "target_2r" : target_2r,
        "risk_pct"  : risk_pct,
        "size_note" : size_note,
    }

# ===================== BACKTESTING =====================
def run_backtest(symbol: str = "TATSILV.NS", days: int = 30) -> dict:
    """
    Simple premium-based backtest over last `days` calendar days.
    Simulates: buy when premium ≤ PREMIUM_BUY, sell when premium ≥ PREMIUM_SELL.
    Returns summary stats.
    """
    print(f"🔍 Running {days}-day backtest on {symbol}...")

    etf_df = fetch_history(symbol,   period=f"{days}d", interval="1d")
    si_df  = fetch_history("SI=F",   period=f"{days}d", interval="1d")
    fx_df  = fetch_history("INR=X",  period=f"{days}d", interval="1d")

    if etf_df is None or si_df is None or fx_df is None:
        return {"error": "Backtest data unavailable"}

    etf   = safe_close(etf_df).rename("etf")
    si    = safe_close(si_df).rename("si")
    fx    = safe_close(fx_df).rename("fx")

    df = pd.concat([etf, si, fx], axis=1).dropna()
    if len(df) < 5:
        return {"error": "Insufficient backtest data"}

    df["inav"]    = df["si"].shift(1) * df["fx"].shift(1) * (1 + (df["si"].pct_change()))
    df["premium"] = (df["etf"] - df["inav"]) / df["inav"] * 100

    # Simulate trades
    trades     = []
    position   = None

    for i, row in df.iterrows():
        p = row["premium"]
        e = row["etf"]
        if np.isnan(p) or np.isnan(e):
            continue

        if position is None and p <= PREMIUM_BUY:
            position = {"entry": e, "date": i}

        elif position is not None and p >= PREMIUM_SELL:
            ret = (e - position["entry"]) / position["entry"] * 100
            trades.append({
                "buy_date"  : str(position["date"])[:10],
                "sell_date" : str(i)[:10],
                "entry"     : round(position["entry"], 2),
                "exit"      : round(e, 2),
                "return_pct": round(ret, 2),
            })
            position = None

    wins       = [t for t in trades if t["return_pct"] > 0]
    total_ret  = sum(t["return_pct"] for t in trades)
    win_rate   = len(wins) / len(trades) * 100 if trades else 0

    return {
        "period_days"  : days,
        "total_trades" : len(trades),
        "win_rate_pct" : round(win_rate, 1),
        "total_return" : round(total_ret, 2),
        "avg_return"   : round(total_ret / len(trades), 2) if trades else 0,
        "trades"       : trades[-5:],   # last 5 trades
    }

# ===================== CSV =====================
FIELDNAMES = [
    "timestamp", "market_phase",
    "silver_global_usd", "gold_global_usd", "usd_inr",
    "etf_price", "gold_etf_price",
    "fair_inav", "premium_pct",
    "rsi", "macd_hist", "bb_pct_b", "atr", "vix",
    "signal", "signal_score", "signal_confidence",
    "stop_loss", "target_1r", "target_2r",
]

def save_csv(row: dict):
    file = Path("silver_history.csv")
    write_header = not file.exists()
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ===================== TELEGRAM =====================
def send_telegram(message: str):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Telegram credentials missing")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id"                  : chat_id,
                    "text"                     : message,
                    "parse_mode"               : "Markdown",
                    "disable_web_page_preview" : True,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                print("📨 Telegram message sent")
                return
            else:
                print(f"⚠️  Telegram attempt {attempt}: {resp.status_code} {resp.text[:120]}")
        except Exception as e:
            print(f"⚠️  Telegram attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            time_module.sleep(RETRY_DELAY)

    print("❌ All Telegram retries failed")

def format_telegram(
    now, market_phase, live, prev,
    curr_inr, fair_inav, premium_pct,
    rsi, macd_line, macd_sig, macd_hist,
    bb_upper, bb_mid, bb_lower, pct_b,
    atr, vix,
    sig: dict, risk: dict,
    bt: dict,
):
    alert_tag = "🚨 *ALERT*" if abs(premium_pct) >= 2 else "📊 *UPDATE*"
    sep = "─" * 28

    # Indicator table (monospace block)
    indicators = (
        "```\n"
        f"{'Indicator':<14} {'Value':>10}\n"
        f"{sep}\n"
        f"{'RSI (15m)':<14} {rsi:>9.1f}\n"
        f"{'MACD Line':<14} {macd_line:>+9.4f}\n"
        f"{'MACD Hist':<14} {macd_hist:>+9.4f}\n"
        f"{'BB %B':<14} {pct_b:>9.2f}\n"
        f"{'ATR':<14} {atr:>9.2f}\n"
        f"{'VIX':<14} {vix:>9.2f}\n"
        "```"
    )

    # Multi-asset prices
    assets = (
        "```\n"
        f"{'Asset':<14} {'Price':>12}\n"
        f"{sep}\n"
        f"{'Silver (oz)':<14} ${live['Silver_Global']:>10.2f}\n"
        f"{'Silver INR':<14} ₹{curr_inr:>10.2f}\n"
        f"{'Gold (oz)':<14} ${live['Gold_Global']:>10.2f}\n"
        f"{'ETF Price':<14} ₹{live['Silver_ETF']:>10.2f}\n"
        f"{'Fair iNAV':<14} ₹{fair_inav:>10.2f}\n"
        f"{'Gold ETF':<14} ₹{live['Gold_ETF']:>10.2f}\n"
        f"{'USD/INR':<14} ₹{live['USD_INR']:>10.4f}\n"
        "```"
    )

    # Risk levels
    risk_block = (
        "```\n"
        f"{'Stop Loss':<14} ₹{risk['stop_loss']:>10.2f}\n"
        f"{'Target 1:1':<14} ₹{risk['target_1r']:>10.2f}\n"
        f"{'Target 1:2':<14} ₹{risk['target_2r']:>10.2f}\n"
        f"{'Risk %':<14} {risk['risk_pct']:>9.2f}%\n"
        "```"
    )

    # Signal reasons
    reason_lines = "\n".join(f"  • {r}" for r in sig["reasons"])

    # Backtest summary
    if "error" not in bt:
        bt_block = (
            f"📈 *30-Day Backtest* ({bt['total_trades']} trades)\n"
            f"Win Rate: {bt['win_rate_pct']}%  |  Avg Return: {bt['avg_return']:+.2f}%"
        )
    else:
        bt_block = f"📈 Backtest: {bt.get('error','N/A')}"

    msg = (
        f"{alert_tag} *Silver Tracker*  {sig['emoji']}\n"
        f"🕒 Phase: *{market_phase}*  |  {now.strftime('%d-%b %H:%M IST')}\n\n"
        f"💰 *Prices*\n{assets}\n"
        f"📐 *Indicators*\n{indicators}\n"
        f"🎯 *Signal: {sig['action']}*  (score {sig['score']:+d}, "
        f"confidence {sig['confidence']:.0f}%)\n"
        f"{reason_lines}\n\n"
        f"🛡 *Risk Management*  [{risk['size_note']}]\n{risk_block}\n"
        f"{bt_block}\n"
        f"_Premium: {premium_pct:+.2f}%_"
    )
    return msg

# ===================== MAIN =====================
def main():
    print("⏳ Running Enhanced Silver Tracker...")

    # ---- Fetch all prices ----
    live, prev = {}, {}
    for k, sym in TICKERS.items():
        lv, pv = get_live_prev(sym)
        live[k], prev[k] = lv, pv
        print(f"  {k}: live={lv:.4f}  prev={pv:.4f}")

    # ---- Intraday data for indicators ----
    raw = fetch_history("TATSILV.NS", period="5d", interval="15m")
    rsi_val = 50.0; macd_line = 0.0; macd_sig = 0.0; macd_hist = 0.0
    bb_upper = 0.0; bb_mid = 0.0; bb_lower = 0.0; pct_b = 0.5
    atr_val = 0.0

    if raw is not None and not raw.empty:
        close_15m = safe_close(raw)
        if len(close_15m) > 26:
            rsi_val             = float(calculate_rsi(close_15m).dropna().iloc[-1])
            macd_line, macd_sig, macd_hist = calculate_macd(close_15m)
            bb_upper, bb_mid, bb_lower, pct_b = calculate_bollinger(close_15m)
        atr_val = calculate_atr(raw)
        print(f"  RSI={rsi_val:.1f}  MACD_hist={macd_hist:+.4f}  BB%B={pct_b:.2f}  ATR={atr_val:.2f}")

    # ---- Silver iNAV ----
    prev_inr   = prev["Silver_Global"] * prev["USD_INR"]
    curr_inr   = live["Silver_Global"] * live["USD_INR"]
    move_pct   = (curr_inr - prev_inr) / prev_inr if prev_inr else 0.0
    fair_inav  = prev["Silver_ETF"] * (1 + move_pct) if prev["Silver_ETF"] else 0.0
    premium_pct = ((live["Silver_ETF"] - fair_inav) / fair_inav * 100) if fair_inav else 0.0

    # ---- VIX ----
    vix = live["India_VIX"] if live["India_VIX"] else 15.0

    # ---- Market Phase ----
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t   = now.time()
    if   t < time(9, 15):  market_phase = "PRE-MARKET"
    elif t > time(15, 30): market_phase = "POST-MARKET"
    else:                  market_phase = "LIVE"

    # ---- Signal ----
    sig  = build_signal(premium_pct, rsi_val, macd_hist, pct_b, vix)
    risk = risk_management(live["Silver_ETF"], atr_val, sig["score"])

    print(f"  Signal: {sig['action']}  score={sig['score']}  confidence={sig['confidence']}%")
    print(f"  Stop: ₹{risk['stop_loss']}  T1: ₹{risk['target_1r']}  T2: ₹{risk['target_2r']}")

    # ---- Backtest (run once per day — skip if today's result already in history) ----
    bt = {}
    bt_cache = Path("backtest_cache.json")
    today_str = now.strftime("%Y-%m-%d")

    if bt_cache.exists():
        try:
            cached = json.loads(bt_cache.read_text())
            if cached.get("date") == today_str:
                bt = cached.get("result", {})
                print("  Backtest: using today's cached result")
        except Exception:
            pass

    if not bt:
        bt = run_backtest(days=30)
        try:
            bt_cache.write_text(json.dumps({"date": today_str, "result": bt}))
        except Exception:
            pass

    if "error" not in bt:
        print(f"  Backtest: {bt['total_trades']} trades, "
              f"win={bt['win_rate_pct']}%, avg={bt['avg_return']:+.2f}%")

    # ---- Save CSV ----
    save_csv({
        "timestamp"          : now.isoformat(),
        "market_phase"       : market_phase,
        "silver_global_usd"  : round(live["Silver_Global"], 4),
        "gold_global_usd"    : round(live["Gold_Global"], 4),
        "usd_inr"            : round(live["USD_INR"], 4),
        "etf_price"          : round(live["Silver_ETF"], 2),
        "gold_etf_price"     : round(live["Gold_ETF"], 2),
        "fair_inav"          : round(fair_inav, 2),
        "premium_pct"        : round(premium_pct, 2),
        "rsi"                : round(rsi_val, 1),
        "macd_hist"          : round(macd_hist, 4),
        "bb_pct_b"           : round(pct_b, 4),
        "atr"                : round(atr_val, 2),
        "vix"                : round(vix, 2),
        "signal"             : sig["action"],
        "signal_score"       : sig["score"],
        "signal_confidence"  : sig["confidence"],
        "stop_loss"          : risk["stop_loss"],
        "target_1r"          : risk["target_1r"],
        "target_2r"          : risk["target_2r"],
    })

    # ---- Telegram ----
    msg = format_telegram(
        now, market_phase, live, prev,
        curr_inr, fair_inav, premium_pct,
        rsi_val, macd_line, macd_sig, macd_hist,
        bb_upper, bb_mid, bb_lower, pct_b,
        atr_val, vix,
        sig, risk, bt,
    )
    send_telegram(msg)

    print("✅ Completed")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
