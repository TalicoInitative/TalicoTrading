import json
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIONS_FILE = Path(__file__).parent / "positions.json"
WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"
HISTORY_FILE = Path(__file__).parent / "trade_history.json"


def get_usd_cad_rate() -> float:
    try:
        t = yf.Ticker("USDCAD=X")
        rate = t.history(period="1d")["Close"].iloc[-1]
        return round(float(rate), 4)
    except Exception:
        return 1.36


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    return df


def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "marketCap": info.get("marketCap"),
            "previousClose": info.get("previousClose"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "targetMedianPrice": info.get("targetMedianPrice"),
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "recommendationKey": info.get("recommendationKey", "none"),
            "recommendationMean": info.get("recommendationMean"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions", 0),
            "shortRatio": info.get("shortRatio"),
            "shortPercentOfFloat": info.get("shortPercentOfFloat"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "earningsTimestamp": info.get("earningsTimestampStart"),
        }
    except Exception:
        return {
            "shortName": ticker, "sector": "Unknown", "marketCap": None,
            "previousClose": None, "targetMeanPrice": None,
            "targetMedianPrice": None, "targetHighPrice": None,
            "targetLowPrice": None, "recommendationKey": "none",
            "recommendationMean": None, "numberOfAnalystOpinions": 0,
            "shortRatio": None, "shortPercentOfFloat": None,
            "beta": None, "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
            "earningsTimestamp": None,
        }


def fetch_analyst_data(ticker: str) -> dict:
    result = {
        "has_data": False, "buy": 0, "hold": 0, "sell": 0,
        "strong_buy": 0, "strong_sell": 0, "total": 0,
        "consensus": "none", "upgrades_downgrades": [],
    }
    try:
        t = yf.Ticker(ticker)
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            result["strong_buy"] = int(latest.get("strongBuy", 0))
            result["buy"] = int(latest.get("buy", 0))
            result["hold"] = int(latest.get("hold", 0))
            result["sell"] = int(latest.get("sell", 0))
            result["strong_sell"] = int(latest.get("strongSell", 0))
            result["total"] = result["strong_buy"] + result["buy"] + result["hold"] + result["sell"] + result["strong_sell"]
            if result["total"] > 0:
                result["has_data"] = True
                bullish = result["strong_buy"] + result["buy"]
                bearish = result["sell"] + result["strong_sell"]
                if bullish > result["total"] * 0.6:
                    result["consensus"] = "bullish"
                elif bearish > result["total"] * 0.4:
                    result["consensus"] = "bearish"
                else:
                    result["consensus"] = "mixed"
    except Exception:
        pass
    try:
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            recent = ud.head(5)
            for _, row in recent.iterrows():
                result["upgrades_downgrades"].append({
                    "date": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
                    "firm": row.get("Firm", "Unknown"),
                    "to_grade": row.get("ToGrade", ""),
                    "from_grade": row.get("FromGrade", ""),
                    "action": row.get("Action", ""),
                })
    except Exception:
        pass
    return result


def fetch_insider_activity(ticker: str) -> dict:
    result = {
        "has_data": False, "net_shares": 0, "net_value": 0,
        "buy_count": 0, "sell_count": 0, "sentiment": "neutral",
        "transactions": [],
    }
    try:
        t = yf.Ticker(ticker)
        txns = t.insider_transactions
        if txns is None or txns.empty:
            return result
        result["has_data"] = True
        cutoff = datetime.now() - timedelta(days=90)
        for _, row in txns.iterrows():
            start_date = row.get("Start Date") or row.get("startDate")
            if start_date is not None:
                if hasattr(start_date, "timestamp"):
                    txn_date = start_date
                else:
                    try:
                        txn_date = pd.Timestamp(start_date)
                    except Exception:
                        continue
                if txn_date.tz_localize(None) if txn_date.tzinfo else txn_date < pd.Timestamp(cutoff):
                    continue
            text = str(row.get("Text", "") or row.get("text", ""))
            shares = abs(float(row.get("Shares", 0) or row.get("shares", 0)))
            value = abs(float(row.get("Value", 0) or row.get("value", 0)))
            is_buy = "purchase" in text.lower() or "buy" in text.lower()
            is_sell = "sale" in text.lower() or "sell" in text.lower()
            if is_buy:
                result["buy_count"] += 1
                result["net_shares"] += shares
                result["net_value"] += value
            elif is_sell:
                result["sell_count"] += 1
                result["net_shares"] -= shares
                result["net_value"] -= value
            result["transactions"].append({
                "insider": str(row.get("Insider", "") or row.get("insider", "")),
                "type": "buy" if is_buy else "sell" if is_sell else "other",
                "shares": shares, "value": value,
            })
        if result["net_value"] > 100_000:
            result["sentiment"] = "bullish"
        elif result["net_value"] < -100_000:
            result["sentiment"] = "bearish"
    except Exception:
        pass
    return result


def fetch_earnings_proximity(info: dict, ticker: str = None) -> dict:
    result = {
        "has_date": False, "days_until": None, "in_swing_window": False, "date_str": None,
        "eps_estimate": None, "revenue_estimate": None,
        "history": [], "avg_surprise_pct": None, "beat_count": 0, "miss_count": 0, "meet_count": 0,
    }
    ts = info.get("earningsTimestamp")
    if ts:
        try:
            if isinstance(ts, (int, float)):
                earn_date = datetime.fromtimestamp(ts).date()
            else:
                earn_date = pd.Timestamp(ts).date()
            days_until = (earn_date - date.today()).days
            if days_until >= 0:
                result["has_date"] = True
                result["days_until"] = days_until
                result["in_swing_window"] = 0 <= days_until <= 21
                result["date_str"] = earn_date.strftime("%Y-%m-%d")
        except Exception:
            pass

    if not ticker:
        return result

    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            today_ts = pd.Timestamp(date.today())
            if ed.index.tz:
                today_ts = today_ts.tz_localize(ed.index.tz)

            future = ed[ed.index >= today_ts].sort_index()
            if not future.empty:
                next_row = future.iloc[0]
                eps_est = next_row.get("EPS Estimate")
                if eps_est is not None and not pd.isna(eps_est):
                    result["eps_estimate"] = round(float(eps_est), 2)
                if not result["has_date"]:
                    next_dt = future.index[0]
                    earn_date = next_dt.date() if hasattr(next_dt, "date") else next_dt
                    days_until = (earn_date - date.today()).days
                    result["has_date"] = True
                    result["date_str"] = earn_date.strftime("%Y-%m-%d")
                    result["days_until"] = days_until
                    result["in_swing_window"] = 0 <= days_until <= 21

            past = ed[ed.index < today_ts].sort_index(ascending=False).head(4)
            surprises = []
            for idx, row in past.iterrows():
                eps_est = row.get("EPS Estimate")
                eps_actual = row.get("Reported EPS")
                surprise_pct = row.get("Surprise(%)")
                if eps_est is not None and not pd.isna(eps_est) and eps_actual is not None and not pd.isna(eps_actual):
                    if surprise_pct is not None and not pd.isna(surprise_pct):
                        s_pct = float(surprise_pct)
                    else:
                        s_pct = ((float(eps_actual) - float(eps_est)) / abs(float(eps_est)) * 100) if float(eps_est) != 0 else 0

                    if s_pct > 1:
                        verdict = "beat"
                        result["beat_count"] += 1
                    elif s_pct < -1:
                        verdict = "miss"
                        result["miss_count"] += 1
                    else:
                        verdict = "met"
                        result["meet_count"] += 1

                    surprises.append(s_pct)
                    result["history"].append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "eps_estimate": round(float(eps_est), 2),
                        "eps_actual": round(float(eps_actual), 2),
                        "surprise_pct": round(s_pct, 1),
                        "verdict": verdict,
                    })

            if surprises:
                result["avg_surprise_pct"] = round(sum(surprises) / len(surprises), 1)
    except Exception:
        pass

    # Revenue estimate from info dict
    rev_est = info.get("revenueEstimate") or info.get("revenueGrowth")
    if rev_est is not None:
        result["revenue_estimate"] = rev_est

    return result


def fetch_news(ticker: str, count: int = 10) -> list[dict]:
    try:
        results = yf.Search(ticker, news_count=count)
        if results.news:
            return results.news[:count]
    except Exception:
        pass
    try:
        news = yf.Ticker(ticker).news
        if news:
            return news[:count]
    except Exception:
        pass
    return []


def detect_support_resistance(df: pd.DataFrame, window: int = 5, num_levels: int = 3):
    recent = df.tail(63) if len(df) > 63 else df
    highs = recent["High"].values
    lows = recent["Low"].values
    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(recent) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            pivot_highs.append(highs[i])
        if lows[i] == min(lows[i - window : i + window + 1]):
            pivot_lows.append(lows[i])

    def cluster(levels, threshold=0.015):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [{"center": levels[0], "count": 1}]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1]["center"]) / clusters[-1]["center"] < threshold:
                c = clusters[-1]
                c["center"] = (c["center"] * c["count"] + lvl) / (c["count"] + 1)
                c["count"] += 1
            else:
                clusters.append({"center": lvl, "count": 1})
        clusters.sort(key=lambda x: x["count"], reverse=True)
        return [round(c["center"], 2) for c in clusters[:num_levels]]

    current_price = df["Close"].iloc[-1]
    support = [l for l in cluster(pivot_lows) if l < current_price]
    resistance = [l for l in cluster(pivot_highs) if l > current_price]

    support.sort(key=lambda x: abs(x - current_price))
    resistance.sort(key=lambda x: abs(x - current_price))

    return support[:num_levels], resistance[:num_levels]


def determine_trend(price, sma_20, sma_50, sma_200):
    if sma_20 is None or sma_50 is None:
        return "insufficient data"
    if sma_200 is not None:
        if price > sma_20 > sma_50 > sma_200:
            return "strong bullish"
        if price < sma_20 < sma_50 < sma_200:
            return "strong bearish"
    if price > sma_20 > sma_50:
        return "bullish"
    if price < sma_20 < sma_50:
        return "bearish"
    return "neutral"


def compute_technical_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    current_price = close.iloc[-1]

    rsi_ind = RSIIndicator(close=close, window=14)
    rsi = rsi_ind.rsi().iloc[-1]
    rsi_prev = rsi_ind.rsi().iloc[-2] if len(rsi_ind.rsi()) >= 2 else rsi
    rsi_5d_ago = rsi_ind.rsi().iloc[-5] if len(rsi_ind.rsi()) >= 5 else rsi

    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]
    macd_hist = macd_ind.macd_diff().iloc[-1]
    macd_hist_prev = macd_ind.macd_diff().iloc[-2] if len(macd_ind.macd_diff()) >= 2 else 0

    macd_series = macd_ind.macd()
    signal_series = macd_ind.macd_signal()
    recent_crossover = False
    crossover_direction = None
    for i in range(-1, max(-6, -len(macd_series)), -1):
        if i - 1 < -len(macd_series):
            break
        curr_diff = macd_series.iloc[i] - signal_series.iloc[i]
        prev_diff = macd_series.iloc[i - 1] - signal_series.iloc[i - 1]
        if not np.isnan(curr_diff) and not np.isnan(prev_diff):
            if curr_diff > 0 and prev_diff <= 0:
                recent_crossover = True
                crossover_direction = "bullish"
                break
            elif curr_diff < 0 and prev_diff >= 0:
                recent_crossover = True
                crossover_direction = "bearish"
                break

    sma_20 = SMAIndicator(close=close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else None
    sma_50 = SMAIndicator(close=close, window=50).sma_indicator().iloc[-1] if len(close) >= 50 else None
    sma_200 = SMAIndicator(close=close, window=200).sma_indicator().iloc[-1] if len(close) >= 200 else None

    ema_12 = EMAIndicator(close=close, window=12).ema_indicator().iloc[-1]
    ema_26 = EMAIndicator(close=close, window=26).ema_indicator().iloc[-1]

    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_middle = bb.bollinger_mavg().iloc[-1]
    bb_width = bb_upper - bb_lower
    bb_position = (current_price - bb_lower) / bb_width if bb_width > 0 else 0.5

    atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
    atr = atr_ind.average_true_range().iloc[-1]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    obv_trend = "rising" if len(obv) >= 10 and obv.iloc[-1] > obv.iloc[-10] else "falling" if len(obv) >= 10 else "flat"

    vol_current = volume.iloc[-1]
    vol_avg_20 = volume.tail(20).mean()
    vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 1.0
    price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0

    support, resistance = detect_support_resistance(df)
    trend = determine_trend(current_price, sma_20, sma_50, sma_200)

    ret_1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) >= 2 else 0
    ret_3d = ((close.iloc[-1] / close.iloc[-4]) - 1) * 100 if len(close) >= 4 else 0
    ret_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0

    consec_up = 0
    consec_down = 0
    for i in range(-1, max(-11, -len(close)), -1):
        if i - 1 < -len(close):
            break
        if close.iloc[i] > close.iloc[i - 1]:
            if consec_down > 0:
                break
            consec_up += 1
        elif close.iloc[i] < close.iloc[i - 1]:
            if consec_up > 0:
                break
            consec_down += 1
        else:
            break

    high_5d = high.tail(5).max() if len(high) >= 5 else high.max()
    low_5d = low.tail(5).min() if len(low) >= 5 else low.min()
    dist_from_5d_high = ((current_price - high_5d) / high_5d) * 100
    dist_from_5d_low = ((current_price - low_5d) / low_5d) * 100

    dist_from_sma20 = ((current_price - sma_20) / sma_20) * 100 if sma_20 else 0
    dist_from_sma50 = ((current_price - sma_50) / sma_50) * 100 if sma_50 else 0

    gap = ((df["Open"].iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100 if len(close) >= 2 else 0

    return {
        "current_price": round(current_price, 2),
        "price_change": round(price_change, 2),
        "price_change_pct": round((price_change / close.iloc[-2]) * 100, 2) if len(close) >= 2 and close.iloc[-2] != 0 else 0,
        "rsi": round(rsi, 2) if not np.isnan(rsi) else 50,
        "rsi_prev": round(rsi_prev, 2) if not np.isnan(rsi_prev) else 50,
        "rsi_5d_ago": round(rsi_5d_ago, 2) if not np.isnan(rsi_5d_ago) else 50,
        "macd_line": round(macd_line, 4),
        "macd_signal": round(macd_signal, 4),
        "macd_hist": round(macd_hist, 4),
        "macd_hist_prev": round(macd_hist_prev, 4),
        "macd_recent_crossover": recent_crossover,
        "macd_crossover_direction": crossover_direction,
        "sma_20": round(sma_20, 2) if sma_20 and not np.isnan(sma_20) else None,
        "sma_50": round(sma_50, 2) if sma_50 and not np.isnan(sma_50) else None,
        "sma_200": round(sma_200, 2) if sma_200 and not np.isnan(sma_200) else None,
        "ema_12": round(ema_12, 2),
        "ema_26": round(ema_26, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_middle": round(bb_middle, 2),
        "bb_position": round(bb_position, 3),
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 2),
        "obv_trend": obv_trend,
        "vol_current": int(vol_current),
        "vol_avg_20": int(vol_avg_20),
        "vol_ratio": round(vol_ratio, 2),
        "price_direction": "up" if price_change > 0 else "down",
        "support_levels": support,
        "resistance_levels": resistance,
        "trend": trend,
        "ret_1d": round(ret_1d, 2),
        "ret_3d": round(ret_3d, 2),
        "ret_5d": round(ret_5d, 2),
        "consec_up": consec_up,
        "consec_down": consec_down,
        "high_5d": round(high_5d, 2),
        "low_5d": round(low_5d, 2),
        "dist_from_5d_high": round(dist_from_5d_high, 2),
        "dist_from_5d_low": round(dist_from_5d_low, 2),
        "dist_from_sma20": round(dist_from_sma20, 2),
        "dist_from_sma50": round(dist_from_sma50, 2),
        "gap_pct": round(gap, 2),
    }


def compute_relative_strength(ticker_df: pd.DataFrame, period: int = 20) -> dict:
    result = {"has_data": False, "rs_ratio": 0, "label": "neutral", "stock_return": 0, "spy_return": 0}
    if len(ticker_df) < period:
        return result
    try:
        spy = yf.Ticker("SPY").history(period="3mo")
        if spy.empty or len(spy) < period:
            return result
        stock_return = (ticker_df["Close"].iloc[-1] / ticker_df["Close"].iloc[-period] - 1) * 100
        spy_return = (spy["Close"].iloc[-1] / spy["Close"].iloc[-period] - 1) * 100
        rs_ratio = stock_return - spy_return
        result["has_data"] = True
        result["rs_ratio"] = round(rs_ratio, 2)
        result["stock_return"] = round(stock_return, 2)
        result["spy_return"] = round(spy_return, 2)
        if rs_ratio > 5:
            result["label"] = "strong outperformer"
        elif rs_ratio > 2:
            result["label"] = "outperformer"
        elif rs_ratio > -2:
            result["label"] = "in line"
        elif rs_ratio > -5:
            result["label"] = "underperformer"
        else:
            result["label"] = "strong underperformer"
    except Exception:
        pass
    return result


def detect_divergences(df: pd.DataFrame) -> list[dict]:
    divergences = []
    if len(df) < 30:
        return divergences
    close = df["Close"]
    rsi_series = RSIIndicator(close=close, window=14).rsi()
    macd_hist_series = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    lookback = min(20, len(df) - 10)

    recent_close = close.iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]

    price_lows = []
    price_highs = []
    for i in range(2, len(recent_close) - 2):
        if recent_close.iloc[i] <= min(recent_close.iloc[i-2:i]) and recent_close.iloc[i] <= min(recent_close.iloc[i+1:i+3]):
            price_lows.append(i)
        if recent_close.iloc[i] >= max(recent_close.iloc[i-2:i]) and recent_close.iloc[i] >= max(recent_close.iloc[i+1:i+3]):
            price_highs.append(i)

    if len(price_lows) >= 2:
        i1, i2 = price_lows[-2], price_lows[-1]
        if recent_close.iloc[i2] < recent_close.iloc[i1] and recent_rsi.iloc[i2] > recent_rsi.iloc[i1]:
            divergences.append({
                "type": "bullish",
                "indicator": "RSI",
                "description": "Price made a lower low but RSI made a higher low — bullish divergence suggests potential reversal up",
            })
    if len(price_highs) >= 2:
        i1, i2 = price_highs[-2], price_highs[-1]
        if recent_close.iloc[i2] > recent_close.iloc[i1] and recent_rsi.iloc[i2] < recent_rsi.iloc[i1]:
            divergences.append({
                "type": "bearish",
                "indicator": "RSI",
                "description": "Price made a higher high but RSI made a lower high — bearish divergence suggests potential reversal down",
            })

    recent_macd = macd_hist_series.iloc[-lookback:]
    if len(price_lows) >= 2:
        i1, i2 = price_lows[-2], price_lows[-1]
        if not np.isnan(recent_macd.iloc[i1]) and not np.isnan(recent_macd.iloc[i2]):
            if recent_close.iloc[i2] < recent_close.iloc[i1] and recent_macd.iloc[i2] > recent_macd.iloc[i1]:
                divergences.append({
                    "type": "bullish",
                    "indicator": "MACD",
                    "description": "Price made a lower low but MACD histogram made a higher low — bullish momentum divergence",
                })
    if len(price_highs) >= 2:
        i1, i2 = price_highs[-2], price_highs[-1]
        if not np.isnan(recent_macd.iloc[i1]) and not np.isnan(recent_macd.iloc[i2]):
            if recent_close.iloc[i2] > recent_close.iloc[i1] and recent_macd.iloc[i2] < recent_macd.iloc[i1]:
                divergences.append({
                    "type": "bearish",
                    "indicator": "MACD",
                    "description": "Price made a higher high but MACD histogram made a lower high — bearish momentum divergence",
                })
    return divergences


def analyze_news_sentiment(news_articles: list[dict]) -> dict:
    sia = SentimentIntensityAnalyzer()
    if not news_articles:
        return {
            "articles": [], "avg_compound": 0, "bullish_count": 0,
            "bearish_count": 0, "neutral_count": 0,
            "overall_sentiment": "neutral", "sentiment_score": 50,
            "has_news": False, "strength": "none",
            "significance": "No recent news available — sentiment defaults to neutral. Check manually for any market-moving events.",
        }

    scored = []
    for article in news_articles:
        title = article.get("title", "")
        if not title:
            continue
        scores = sia.polarity_scores(title)
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "bullish"
        elif compound <= -0.05:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        scored.append({
            "title": title,
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "compound_score": round(compound, 3),
            "sentiment": sentiment,
        })

    if not scored:
        return {
            "articles": [], "avg_compound": 0, "bullish_count": 0,
            "bearish_count": 0, "neutral_count": 0,
            "overall_sentiment": "neutral", "sentiment_score": 50,
            "has_news": False, "strength": "none",
            "significance": "No scorable headlines found.",
        }

    avg_compound = sum(a["compound_score"] for a in scored) / len(scored)
    bullish = sum(1 for a in scored if a["sentiment"] == "bullish")
    bearish = sum(1 for a in scored if a["sentiment"] == "bearish")
    neutral = sum(1 for a in scored if a["sentiment"] == "neutral")

    if avg_compound >= 0.05:
        overall = "bullish"
    elif avg_compound <= -0.05:
        overall = "bearish"
    else:
        overall = "neutral"

    avg_intensity = sum(abs(a["compound_score"]) for a in scored) / len(scored)
    if len(scored) >= 5 and avg_intensity > 0.3:
        strength = "strong"
    elif len(scored) >= 3 or avg_intensity > 0.15:
        strength = "moderate"
    else:
        strength = "weak"

    if overall == "bullish":
        if strength == "strong":
            significance = (f"{bullish} bullish headlines with strong conviction — sustained positive coverage "
                           f"often precedes 1-2 week momentum in swing trades. High-confidence sentiment signal.")
        elif strength == "moderate":
            significance = (f"{bullish} bullish headlines — positive news flow supports the setup, "
                           f"but conviction is moderate. Watch for follow-through.")
        else:
            significance = (f"Mildly bullish sentiment across {len(scored)} headlines — "
                           f"not enough coverage to drive meaningful momentum alone.")
    elif overall == "bearish":
        if strength == "strong":
            significance = (f"{bearish} bearish headlines with strong negative sentiment — "
                           f"elevated negative coverage can accelerate selling pressure over 1-2 weeks. Caution for new entries.")
        elif strength == "moderate":
            significance = (f"{bearish} bearish headlines — negative coverage may weigh on price. "
                           f"Consider waiting for sentiment to stabilize before entering.")
        else:
            significance = (f"Mildly bearish sentiment across {len(scored)} headlines — "
                           f"not strongly directional but leans negative.")
    else:
        significance = (f"Mixed or neutral sentiment across {len(scored)} headlines "
                       f"({bullish} bullish, {bearish} bearish, {neutral} neutral) — "
                       f"sentiment is not a strong factor in this setup.")

    return {
        "articles": scored,
        "avg_compound": round(avg_compound, 3),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "overall_sentiment": overall,
        "sentiment_score": round((avg_compound + 1) * 50, 1),
        "has_news": True,
        "strength": strength,
        "significance": significance,
    }


# --- Scoring functions ---

def _score_rsi(rsi: float) -> tuple[float, str]:
    if rsi < 30:
        score = 80 + (30 - rsi) * (20 / 30)
        reason = f"RSI at {rsi} — oversold territory, strong swing entry window"
    elif rsi < 35:
        score = 70 + (35 - rsi) * 2
        reason = f"RSI at {rsi} — approaching oversold, potential entry developing"
    elif rsi < 45:
        score = 55 + (45 - rsi) * 1.5
        reason = f"RSI at {rsi} — leaning bullish for momentum"
    elif rsi <= 55:
        score = 50
        reason = f"RSI at {rsi} — neutral momentum"
    elif rsi <= 65:
        score = 50 - (rsi - 55) * 1.5
        reason = f"RSI at {rsi} — leaning overbought"
    elif rsi <= 70:
        score = 30 - (rsi - 65) * 2
        reason = f"RSI at {rsi} — approaching overbought, caution for new entries"
    else:
        score = max(0, 20 - (rsi - 70) * (20 / 30))
        reason = f"RSI at {rsi} — overbought, risky entry for a 1-3 week hold"
    return round(min(100, max(0, score)), 1), reason


def _score_macd(hist, hist_prev, recent_crossover, crossover_dir):
    if recent_crossover and crossover_dir == "bullish":
        score = 85
        reason = "Fresh bullish MACD crossover — strong entry signal for swing trades"
    elif recent_crossover and crossover_dir == "bearish":
        score = 15
        reason = "Fresh bearish MACD crossover — momentum turning against longs"
    elif hist > 0 and hist > hist_prev:
        score = 80
        reason = "MACD histogram positive and expanding — bullish momentum building"
    elif hist > 0 and hist <= hist_prev:
        score = 60
        reason = "MACD histogram positive but contracting — bullish momentum fading"
    elif hist < 0 and hist > hist_prev:
        score = 40
        reason = "MACD histogram negative but converging — bearish momentum weakening"
    elif hist < 0 and hist <= hist_prev:
        score = 15
        reason = "MACD histogram negative and expanding — bearish momentum building"
    else:
        score = 50
        reason = "MACD at neutral crossover point"
    return round(score, 1), reason


def _score_moving_averages(price, sma_20, sma_50, sma_200):
    if sma_20 is None or sma_50 is None:
        return 50, "Insufficient data for moving average analysis"
    if sma_200 is not None:
        if price > sma_20 > sma_50 > sma_200:
            return 95, "All MAs aligned bullish (price > SMA20 > SMA50 > SMA200) — strong uptrend"
        if price < sma_20 < sma_50 < sma_200:
            return 5, "All MAs aligned bearish (price < SMA20 < SMA50 < SMA200) — strong downtrend"
        if price > sma_20 > sma_50 and price > sma_200:
            return 80, "Price above all key MAs with SMA20 > SMA50 — bullish swing setup"
        if price < sma_20 < sma_50 and price < sma_200:
            return 20, "Price below all key MAs with SMA20 < SMA50 — bearish"
        if price > sma_200:
            return 55, "Price above SMA200 (bullish bias) but mixed shorter-term alignment"
        return 40, "Price below SMA200 (bearish bias) with mixed shorter-term alignment"
    if price > sma_20 > sma_50:
        return 80, "Price above SMA20 and SMA50 in bullish alignment — good swing setup"
    if price < sma_20 < sma_50:
        return 20, "Price below SMA20 and SMA50 in bearish alignment"
    if price > sma_20:
        return 55, "Price above SMA20 but mixed SMA50 relationship"
    return 40, "Price below SMA20 — near-term trend is down"


def _score_volume(vol_ratio, price_direction):
    if vol_ratio > 1.5 and price_direction == "up":
        return 90, f"Volume {vol_ratio:.1f}x average on up day — strong conviction buying"
    elif vol_ratio > 1.2 and price_direction == "up":
        return 70, f"Volume {vol_ratio:.1f}x average on up day — above-average interest"
    elif vol_ratio > 1.5 and price_direction == "down":
        return 15, f"Volume {vol_ratio:.1f}x average on down day — heavy selling pressure"
    elif vol_ratio > 1.2 and price_direction == "down":
        return 30, f"Volume {vol_ratio:.1f}x average on down day — elevated selling"
    return 50, f"Volume near average ({vol_ratio:.1f}x) — no strong conviction signal"


def _score_bollinger(bb_position):
    if bb_position < 0.1:
        return 85, "Price near lower Bollinger Band — potential bounce entry for swing"
    elif bb_position < 0.3:
        return 65, "Price in lower half of Bollinger Bands — room to run up"
    elif bb_position < 0.5:
        return 55, "Price in lower-mid Bollinger range"
    elif bb_position < 0.7:
        return 45, "Price in upper-mid Bollinger range"
    elif bb_position < 0.9:
        return 30, "Price in upper Bollinger range — limited upside before pullback"
    return 15, "Price near upper Bollinger Band — likely to pull back, risky entry"


def _score_analyst_target(current_price, mean_target, analyst_count):
    if analyst_count is None or analyst_count < 3 or mean_target is None:
        return 50, "Insufficient analyst coverage — defaulting to neutral"
    upside = ((mean_target - current_price) / current_price) * 100
    if upside > 20:
        return 90, f"Price {upside:.0f}% below mean analyst target — deep value vs consensus"
    elif upside > 10:
        return 75, f"Price {upside:.0f}% below mean analyst target — meaningful upside to consensus"
    elif upside > 0:
        return 55, f"Price {upside:.0f}% below mean analyst target — modest upside"
    elif upside > -10:
        return 35, f"Price {abs(upside):.0f}% above mean analyst target — near fair value"
    return 15, f"Price {abs(upside):.0f}% above mean analyst target — consensus says overvalued"


def _score_relative_strength(rs_data):
    if not rs_data.get("has_data"):
        return 50, "Relative strength data unavailable — defaulting to neutral"
    rs = rs_data["rs_ratio"]
    if rs > 5:
        return 85, f"Outperforming SPY by {rs:.1f}% over 20 days — strong relative momentum"
    elif rs > 2:
        return 70, f"Outperforming SPY by {rs:.1f}% — positive relative strength"
    elif rs > -2:
        return 50, f"Roughly in line with SPY ({rs:+.1f}%) — no relative edge"
    elif rs > -5:
        return 30, f"Underperforming SPY by {abs(rs):.1f}% — weak relative strength"
    return 15, f"Underperforming SPY by {abs(rs):.1f}% — significant relative weakness"


def _score_insider_activity(insider_data):
    if not insider_data.get("has_data"):
        return 50, "No insider transaction data available — defaulting to neutral"
    nv = insider_data["net_value"]
    bc = insider_data["buy_count"]
    sc = insider_data["sell_count"]
    if nv > 1_000_000:
        return 85, f"Net insider buying of ${nv/1e6:.1f}M in 90 days ({bc} buys) — strong confidence signal"
    elif nv > 100_000:
        return 70, f"Net insider buying of ${nv/1e3:.0f}K in 90 days ({bc} buys) — positive insider sentiment"
    elif nv > -100_000:
        return 50, f"Minimal insider activity ({bc} buys, {sc} sells) — no strong signal"
    elif nv > -1_000_000:
        return 35, f"Net insider selling of ${abs(nv)/1e3:.0f}K in 90 days ({sc} sells) — caution"
    return 20, f"Net insider selling of ${abs(nv)/1e6:.1f}M in 90 days ({sc} sells) — significant selling pressure"


def compute_rating(technicals, sentiment, analyst_data=None, rs_data=None, insider_data=None, info=None, divergences=None):
    rsi_score, rsi_reason = _score_rsi(technicals["rsi"])
    macd_score, macd_reason = _score_macd(
        technicals["macd_hist"], technicals["macd_hist_prev"],
        technicals["macd_recent_crossover"], technicals["macd_crossover_direction"]
    )
    ma_score, ma_reason = _score_moving_averages(
        technicals["current_price"], technicals["sma_20"],
        technicals["sma_50"], technicals["sma_200"]
    )
    vol_score, vol_reason = _score_volume(technicals["vol_ratio"], technicals["price_direction"])
    bb_score, bb_reason = _score_bollinger(technicals["bb_position"])

    sent_score = sentiment["sentiment_score"]
    if not sentiment["has_news"]:
        sent_reason = "News sentiment not available — defaulting to neutral"
    else:
        overall = sentiment["overall_sentiment"]
        b = sentiment["bullish_count"]
        br = sentiment["bearish_count"]
        n = sentiment["neutral_count"]
        strength = sentiment.get("strength", "")
        sent_reason = (f"News sentiment {overall} ({b} bullish, {br} bearish, {n} neutral) — "
                      f"{strength} signal. " + sentiment.get("significance", ""))

    current_price = technicals["current_price"]
    mean_target = info.get("targetMeanPrice") if info else None
    analyst_count = info.get("numberOfAnalystOpinions") if info else None
    at_score, at_reason = _score_analyst_target(current_price, mean_target, analyst_count)

    rs_score, rs_reason = _score_relative_strength(rs_data or {})
    ins_score, ins_reason = _score_insider_activity(insider_data or {})

    combined = (
        ma_score * 0.18
        + rsi_score * 0.14
        + macd_score * 0.14
        + at_score * 0.12
        + rs_score * 0.12
        + sent_score * 0.10
        + vol_score * 0.08
        + bb_score * 0.07
        + ins_score * 0.05
    )

    if divergences:
        bullish_div = any(d["type"] == "bullish" for d in divergences)
        bearish_div = any(d["type"] == "bearish" for d in divergences)
        if bullish_div and not bearish_div:
            combined += 10
        elif bearish_div and not bullish_div:
            combined -= 10

    combined = round(min(100, max(0, combined)), 1)

    if combined >= 80:
        rating = "Strong Buy"
    elif combined >= 65:
        rating = "Buy"
    elif combined >= 45:
        rating = "Neutral"
    elif combined >= 30:
        rating = "Sell"
    else:
        rating = "Strong Sell"

    components = {
        "moving_averages": {"score": ma_score, "weight": 0.18, "reasoning": ma_reason},
        "rsi": {"score": rsi_score, "weight": 0.14, "reasoning": rsi_reason},
        "macd": {"score": macd_score, "weight": 0.14, "reasoning": macd_reason},
        "analyst_target": {"score": at_score, "weight": 0.12, "reasoning": at_reason},
        "relative_strength": {"score": rs_score, "weight": 0.12, "reasoning": rs_reason},
        "sentiment": {"score": sent_score, "weight": 0.10, "reasoning": sent_reason},
        "volume": {"score": vol_score, "weight": 0.08, "reasoning": vol_reason},
        "bollinger": {"score": bb_score, "weight": 0.07, "reasoning": bb_reason},
        "insider_activity": {"score": ins_score, "weight": 0.05, "reasoning": ins_reason},
    }

    deviations = [(name, abs(c["score"] - 50), c["reasoning"]) for name, c in components.items()]
    deviations.sort(key=lambda x: x[1], reverse=True)
    key_signals = [d[2] for d in deviations[:3]]

    if divergences:
        for d in divergences:
            key_signals.insert(0, f"DIVERGENCE: {d['description']}")

    return {
        "combined_score": combined,
        "rating": rating,
        "component_scores": components,
        "key_signals": key_signals[:5],
    }


COMPONENT_LABELS = {
    "moving_averages": "MA Alignment",
    "rsi": "RSI",
    "macd": "MACD",
    "analyst_target": "Analyst Target",
    "relative_strength": "Relative Strength",
    "sentiment": "Sentiment",
    "volume": "Volume",
    "bollinger": "Bollinger",
    "insider_activity": "Insider Activity",
}


def explain_rating_change(prev_day, curr_day):
    if prev_day is None:
        return {
            "summary": "Baseline — first day in the lookback window.",
            "details": [],
        }
    prev_score = prev_day["score"]
    curr_score = curr_day["score"]
    diff = curr_score - prev_score
    prev_comps = prev_day.get("component_scores", {})
    curr_comps = curr_day.get("component_scores", {})

    deltas = []
    for key in curr_comps:
        if key in prev_comps:
            cs = curr_comps[key]["score"]
            ps = prev_comps[key]["score"]
            d = cs - ps
            if abs(d) >= 3:
                label = COMPONENT_LABELS.get(key, key.replace("_", " ").title())
                reasoning = curr_comps[key].get("reasoning", "")
                deltas.append({"label": label, "delta": d, "prev_score": ps,
                               "curr_score": cs, "reasoning": reasoning, "key": key})

    deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)

    if abs(diff) < 1:
        direction = "Score unchanged"
    elif diff > 0:
        direction = f"Score rose {diff:+.0f}"
    else:
        direction = f"Score fell {diff:+.0f}"

    prev_rating = prev_day.get("rating", "")
    curr_rating = curr_day.get("rating", "")
    if prev_rating != curr_rating and prev_rating and curr_rating:
        direction += f" — rating changed from {prev_rating} to {curr_rating}"

    short_parts = []
    for d in deltas[:3]:
        short_parts.append(f"{d['label']} {d['delta']:+.0f}")
    summary = f"{direction}: {', '.join(short_parts)}." if short_parts else f"{direction}."

    details = []
    for d in deltas[:5]:
        arrow = "improved" if d["delta"] > 0 else "weakened"
        detail = f"{d['label']} {arrow} ({d['prev_score']:.0f} → {d['curr_score']:.0f}, {d['delta']:+.0f} pts)"
        if d["reasoning"]:
            detail += f" — {d['reasoning']}"
        details.append({"text": detail, "delta": d["delta"], "label": d["label"]})

    return {"summary": summary, "details": details}


def compute_rating_history(df, sentiment, lookback_days=5, analyst_data=None, rs_data=None, insider_data=None, info=None):
    history = []
    for offset in range(lookback_days, 0, -1):
        if len(df) <= offset + 26:
            continue
        sliced = df.iloc[:-offset]
        try:
            tech = compute_technical_indicators(sliced)
            r = compute_rating(tech, sentiment, analyst_data=analyst_data,
                               rs_data=rs_data, insider_data=insider_data, info=info)
            trade_date = sliced.index[-1]
            history.append({
                "date": trade_date.strftime("%Y-%m-%d"),
                "score": r["combined_score"],
                "rating": r["rating"],
                "rsi": tech["rsi"],
                "macd_hist": tech["macd_hist"],
                "trend": tech["trend"],
                "component_scores": r["component_scores"],
            })
        except Exception:
            pass

    for i, day in enumerate(history):
        prev = history[i - 1] if i > 0 else None
        day["change_explanation"] = explain_rating_change(prev, day)

    return history


def compute_buy_timing(technicals, sentiment, rating_data, earnings, divergences, info, rs_data):
    t = technicals
    rsi = t["rsi"]
    score = rating_data["combined_score"]
    reasons = []
    risk_factors = []
    better_entry = []

    buy_signals = 0
    caution_signals = 0

    if rsi < 35:
        buy_signals += 2
        reasons.append(f"RSI at {rsi:.0f} — oversold, historically strong entry zone")
    elif rsi < 45:
        buy_signals += 1
        reasons.append(f"RSI at {rsi:.0f} — neutral-low, room to run")
    elif rsi > 70:
        caution_signals += 2
        risk_factors.append(f"RSI at {rsi:.0f} — overbought, pullback likely before next leg up")
        better_entry.append("Wait for RSI to cool below 60")
    elif rsi > 60:
        caution_signals += 1
        risk_factors.append(f"RSI at {rsi:.0f} — elevated, not ideal for fresh entries")

    if t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bullish":
        buy_signals += 2
        reasons.append("Fresh bullish MACD crossover — momentum just turned positive")
    elif t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bearish":
        caution_signals += 2
        risk_factors.append("Fresh bearish MACD crossover — momentum turning negative")
    elif t["macd_hist"] > 0 and t["macd_hist"] > t["macd_hist_prev"]:
        buy_signals += 1
        reasons.append("MACD histogram expanding bullish — momentum building")
    elif t["macd_hist"] < 0 and t["macd_hist"] < t["macd_hist_prev"]:
        caution_signals += 1
        risk_factors.append("MACD histogram expanding bearish — selling pressure increasing")

    if "bullish" in t["trend"]:
        buy_signals += 1
        reasons.append(f"Trend is {t['trend']} — price above key moving averages")
    elif "bearish" in t["trend"]:
        caution_signals += 2
        risk_factors.append(f"Trend is {t['trend']} — fighting the primary trend")

    if t["support_levels"]:
        nearest_sup = t["support_levels"][0]
        sup_dist = ((t["current_price"] - nearest_sup) / t["current_price"]) * 100
        if sup_dist < 2:
            buy_signals += 2
            reasons.append(f"Price within 2% of support at ${nearest_sup} — tight stop possible, good risk/reward")
        elif sup_dist < 5:
            buy_signals += 1
            reasons.append(f"Support at ${nearest_sup} ({sup_dist:.1f}% below) — reasonable stop-loss level")
        if sup_dist > 8:
            caution_signals += 1
            risk_factors.append(f"Nearest support is {sup_dist:.1f}% below — wide stop required, poor risk/reward")
            better_entry.append(f"Wait for a pullback toward support at ${nearest_sup}")

    if t["resistance_levels"]:
        nearest_res = t["resistance_levels"][0]
        res_dist = ((nearest_res - t["current_price"]) / t["current_price"]) * 100
        if res_dist < 1.5:
            caution_signals += 1
            risk_factors.append(f"Resistance at ${nearest_res} is only {res_dist:.1f}% above — limited upside before resistance")
            better_entry.append("Wait for a breakout above resistance or a pullback to support")

    if t["vol_ratio"] > 1.3 and t["price_direction"] == "up":
        buy_signals += 1
        reasons.append(f"Volume {t['vol_ratio']:.1f}x average on an up day — buyers stepping in")
    elif t["vol_ratio"] > 1.3 and t["price_direction"] == "down":
        caution_signals += 1
        risk_factors.append(f"Volume {t['vol_ratio']:.1f}x average on a down day — distribution")

    if abs(t.get("dist_from_sma20", 0)) > 5:
        if t["dist_from_sma20"] > 5:
            caution_signals += 1
            risk_factors.append(f"Price {t['dist_from_sma20']:.1f}% above SMA20 — overextended, mean-reversion risk")
            better_entry.append("Wait for price to pull back toward the 20-day moving average")
        elif t["dist_from_sma20"] < -5:
            buy_signals += 1
            reasons.append(f"Price {abs(t['dist_from_sma20']):.1f}% below SMA20 — stretched to downside, bounce likely")

    if "bullish" in t["trend"] and t.get("ret_5d", 0) < -3 and rsi < 50:
        buy_signals += 1
        reasons.append(f"Pullback of {abs(t['ret_5d']):.1f}% in a bullish trend — classic pullback entry")

    if t.get("consec_up", 0) >= 5:
        caution_signals += 1
        risk_factors.append(f"{t['consec_up']} consecutive up days — short-term exhaustion risk")
        better_entry.append("Wait for at least 1-2 red days to reset short-term momentum")
    if t.get("consec_down", 0) >= 4 and rsi < 40:
        buy_signals += 1
        reasons.append(f"{t['consec_down']} consecutive down days with RSI at {rsi:.0f} — washout may be ending")

    if sentiment.get("overall_sentiment") == "bullish" and sentiment.get("strength") in ("strong", "moderate"):
        buy_signals += 1
        reasons.append("Positive news sentiment supports the setup")
    elif sentiment.get("overall_sentiment") == "bearish" and sentiment.get("strength") in ("strong", "moderate"):
        caution_signals += 1
        risk_factors.append("Negative news sentiment — headwinds for any long position")

    if earnings.get("has_date") and earnings.get("in_swing_window"):
        days_to = earnings["days_until"]
        if days_to <= 5:
            caution_signals += 2
            risk_factors.append(f"Earnings in {days_to} days — binary event risk, gap risk overnight")
            better_entry.append("Wait until after earnings to avoid gap risk")
        elif days_to <= 14:
            caution_signals += 1
            risk_factors.append(f"Earnings in {days_to} days — position sizing should be reduced")

    if divergences:
        for d in divergences:
            if d["type"] == "bullish":
                buy_signals += 1
                reasons.append(f"Bullish {d['indicator']} divergence detected — reversal signal")
            else:
                caution_signals += 1
                risk_factors.append(f"Bearish {d['indicator']} divergence — potential reversal down")

    if rs_data and rs_data.get("has_data") and rs_data["rs_ratio"] > 5:
        buy_signals += 1
        reasons.append(f"Outperforming SPY by {rs_data['rs_ratio']:.1f}% — strong relative strength")

    net = buy_signals - caution_signals

    if net >= 4 and score >= 65:
        timing = "Buy Now"
        confidence = "high"
    elif net >= 3 and score >= 55:
        timing = "Buy Now"
        confidence = "moderate"
    elif "bullish" in t["trend"] and t.get("ret_5d", 0) < -2 and rsi < 50 and net >= 1:
        timing = "Buy — Pullback Entry"
        confidence = "moderate"
    elif net >= 1 and score >= 50:
        timing = "Watch for Entry"
        confidence = "low"
    elif rsi > 70 or t.get("dist_from_sma20", 0) > 5:
        timing = "Overextended"
        confidence = "moderate"
    elif "bullish" in t["trend"] and rsi > 55 and net < 2:
        timing = "Wait for Pullback"
        confidence = "moderate"
        if not better_entry:
            better_entry.append("Wait for RSI to drop below 50 or price to touch SMA20")
    elif caution_signals >= 3 or score < 35:
        timing = "Avoid for Now"
        confidence = "high" if caution_signals >= 4 else "moderate"
    elif net <= -1:
        timing = "Risky Entry"
        confidence = "moderate"
    else:
        timing = "Watch for Entry"
        confidence = "low"

    if not better_entry and timing not in ("Buy Now", "Buy — Pullback Entry"):
        if rsi > 50:
            better_entry.append("Wait for RSI to drop below 45")
        if t["support_levels"]:
            better_entry.append(f"Wait for price to pull back toward support at ${t['support_levels'][0]}")
        if t["macd_hist"] < 0:
            better_entry.append("Wait for MACD to cross bullish")

    return {
        "timing": timing,
        "confidence": confidence,
        "buy_signals": buy_signals,
        "caution_signals": caution_signals,
        "reasons": reasons[:6],
        "risk_factors": risk_factors[:5],
        "better_entry": better_entry[:3],
    }


def generate_written_analysis(ticker, technicals, sentiment, rating_data, buy_timing,
                               earnings, analyst, insider, rs_data, divergences, info):
    t = technicals
    bt = buy_timing
    rsi = t["rsi"]
    price = t["current_price"]
    score = rating_data["combined_score"]
    timing = bt["timing"]

    summary_parts = []
    if timing == "Buy Now":
        summary_parts.append(f"{ticker} is showing a strong entry setup right now.")
    elif timing == "Buy — Pullback Entry":
        summary_parts.append(f"{ticker} is pulling back within an uptrend — this could be a good entry.")
    elif timing == "Watch for Entry":
        summary_parts.append(f"{ticker} has some positive signals but isn't a clear buy yet.")
    elif timing == "Overextended":
        summary_parts.append(f"{ticker} has run too far too fast — wait for a pullback before entering.")
    elif timing == "Wait for Pullback":
        summary_parts.append(f"{ticker} is in a good trend but needs to cool off before entering.")
    elif timing == "Risky Entry":
        summary_parts.append(f"{ticker} has mixed signals — buying here carries above-average risk.")
    else:
        summary_parts.append(f"{ticker} is not in a favorable position for a swing trade entry right now.")

    summary_parts.append(f"The stock is trading at ${price:.2f} with an overall score of {score:.0f}/100.")

    trend_desc = t["trend"]
    if "strong bullish" in trend_desc:
        summary_parts.append("All major moving averages are aligned bullish — the primary trend is strongly up.")
    elif "bullish" in trend_desc:
        summary_parts.append("The short-term trend is bullish with price above key moving averages.")
    elif "strong bearish" in trend_desc:
        summary_parts.append("All major moving averages are aligned bearish — this is a downtrend.")
    elif "bearish" in trend_desc:
        summary_parts.append("The short-term trend is bearish — price is below key moving averages.")
    else:
        summary_parts.append("The trend is neutral — no clear directional bias from moving averages.")

    bull_signals = []
    bear_signals = []

    if rsi < 35:
        bull_signals.append(f"RSI is oversold at {rsi:.0f} — historically a high-probability reversal zone")
    elif rsi < 45:
        bull_signals.append(f"RSI at {rsi:.0f} is in neutral-low territory — room for upside")
    if rsi > 70:
        bear_signals.append(f"RSI is overbought at {rsi:.0f} — elevated risk of a pullback")
    elif rsi > 60:
        bear_signals.append(f"RSI at {rsi:.0f} is approaching overbought — momentum getting extended")

    if t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bullish":
        bull_signals.append("MACD just crossed bullish — fresh momentum shift to the upside")
    elif t["macd_hist"] > 0 and t["macd_hist"] > t["macd_hist_prev"]:
        bull_signals.append("MACD histogram is positive and expanding — bullish momentum building")
    if t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bearish":
        bear_signals.append("MACD just crossed bearish — momentum shifting against longs")
    elif t["macd_hist"] < 0 and t["macd_hist"] < t["macd_hist_prev"]:
        bear_signals.append("MACD histogram is negative and expanding — bearish momentum increasing")

    if "bullish" in trend_desc:
        bull_signals.append(f"Price is in a {trend_desc} trend — MAs are lined up for upside")
    if "bearish" in trend_desc:
        bear_signals.append(f"Price is in a {trend_desc} trend — fighting the trend is risky")

    if t["vol_ratio"] > 1.3 and t["price_direction"] == "up":
        bull_signals.append(f"Volume is {t['vol_ratio']:.1f}x average on an up day — institutional interest")
    if t["vol_ratio"] > 1.3 and t["price_direction"] == "down":
        bear_signals.append(f"Volume is {t['vol_ratio']:.1f}x average on a down day — distribution selling")

    if t["bb_position"] < 0.15:
        bull_signals.append("Price near lower Bollinger Band — mean-reversion bounce likely")
    if t["bb_position"] > 0.85:
        bear_signals.append("Price near upper Bollinger Band — overextended, pullback risk")

    if sentiment.get("overall_sentiment") == "bullish":
        bull_signals.append(f"News sentiment is bullish ({sentiment.get('strength', '')} strength)")
    elif sentiment.get("overall_sentiment") == "bearish":
        bear_signals.append(f"News sentiment is bearish ({sentiment.get('strength', '')} strength)")

    if rs_data and rs_data.get("has_data"):
        if rs_data["rs_ratio"] > 3:
            bull_signals.append(f"Outperforming SPY by {rs_data['rs_ratio']:.1f}% — relative strength leader")
        elif rs_data["rs_ratio"] < -3:
            bear_signals.append(f"Underperforming SPY by {abs(rs_data['rs_ratio']):.1f}% — relative weakness")

    if divergences:
        for d in divergences:
            if d["type"] == "bullish":
                bull_signals.append(f"Bullish {d['indicator']} divergence — price weakness not confirmed by momentum")
            else:
                bear_signals.append(f"Bearish {d['indicator']} divergence — price strength not confirmed by momentum")

    if info.get("targetMeanPrice") and info.get("numberOfAnalystOpinions", 0) >= 3:
        upside = ((info["targetMeanPrice"] - price) / price) * 100
        if upside > 10:
            bull_signals.append(f"Analyst mean target implies {upside:.0f}% upside")
        elif upside < -5:
            bear_signals.append(f"Analyst mean target implies {abs(upside):.0f}% downside — consensus says overvalued")

    risks = []
    if earnings.get("has_date") and earnings.get("in_swing_window"):
        risks.append(f"Earnings report in {earnings['days_until']} days ({earnings['date_str']}) — binary event with gap risk")
    if rsi > 65:
        risks.append("RSI is elevated — a pullback could erase entry gains quickly")
    if t.get("dist_from_sma20", 0) > 5:
        risks.append(f"Price is {t['dist_from_sma20']:.1f}% above SMA20 — mean-reversion risk")
    if "bearish" in trend_desc:
        risks.append("Buying against the trend — higher probability of the trade going against you")
    if sentiment.get("overall_sentiment") == "bearish" and sentiment.get("strength") == "strong":
        risks.append("Strong negative news flow could accelerate selling")
    if t.get("atr_pct", 0) > 4:
        risks.append(f"High volatility (ATR {t['atr_pct']:.1f}% of price) — wider stops needed, more pain on drawdowns")
    if insider and insider.get("sentiment") == "bearish":
        risks.append("Insiders have been net sellers — they know the company better than anyone")

    if score >= 70:
        short_term = "Bullish — multiple signals confirm buying pressure. Expect continuation if volume holds."
    elif score >= 55:
        short_term = "Cautiously bullish — setup is developing but needs confirmation from volume or a breakout."
    elif score >= 45:
        short_term = "Neutral — no strong directional bias. Could go either way in the next 1-2 weeks."
    elif score >= 30:
        short_term = "Cautiously bearish — more selling pressure than buying. Avoid longs unless clear reversal signal."
    else:
        short_term = "Bearish — selling pressure dominant. Not a good environment for swing trade entries."

    if score >= 65 and "bullish" in trend_desc:
        medium_term = "Positive — uptrend intact with strong internals. Favorable for 2-4 week holds."
    elif score >= 50:
        medium_term = "Mixed — some positive elements but the trend needs to prove itself. Watch for breakout or breakdown."
    else:
        medium_term = "Negative — trend and momentum favor the downside. Wait for a base to form before considering entries."

    if bt["confidence"] == "high":
        conf_text = "High — multiple independent signals confirm this assessment"
    elif bt["confidence"] == "moderate":
        conf_text = "Moderate — signal is present but not fully confirmed across all indicators"
    else:
        conf_text = "Low — signals are mixed, proceed with caution and smaller position size"

    return {
        "summary": " ".join(summary_parts),
        "bull_signals": bull_signals[:6],
        "bear_signals": bear_signals[:6],
        "risks": risks[:5],
        "short_term_outlook": short_term,
        "medium_term_outlook": medium_term,
        "confidence": conf_text,
        "better_entry": bt["better_entry"],
    }


def analyze_ticker(ticker: str, period: str = "6mo") -> dict:
    try:
        df = fetch_stock_data(ticker, period)
        info = fetch_ticker_info(ticker)
        technicals = compute_technical_indicators(df)
        news = fetch_news(ticker)
        sentiment = analyze_news_sentiment(news)
        analyst = fetch_analyst_data(ticker)
        insider = fetch_insider_activity(ticker)
        earnings = fetch_earnings_proximity(info, ticker=ticker)
        rs_data = compute_relative_strength(df)
        divergences = detect_divergences(df)

        rating = compute_rating(technicals, sentiment,
                                analyst_data=analyst, rs_data=rs_data,
                                insider_data=insider, info=info,
                                divergences=divergences)

        rating_history = compute_rating_history(df, sentiment, lookback_days=5,
                                                analyst_data=analyst, rs_data=rs_data,
                                                insider_data=insider, info=info)

        sent_dir = sentiment["overall_sentiment"]
        tech_dir = "bullish" if "bullish" in technicals["trend"] else "bearish" if "bearish" in technicals["trend"] else "neutral"
        if sent_dir == tech_dir and sent_dir != "neutral":
            agreement = "aligned"
        elif sent_dir != "neutral" and tech_dir != "neutral" and sent_dir != tech_dir:
            agreement = "divergent"
        else:
            agreement = "neutral"

        buy_timing = compute_buy_timing(technicals, sentiment, rating, earnings,
                                         divergences, info, rs_data)
        written = generate_written_analysis(ticker.upper(), technicals, sentiment, rating,
                                             buy_timing, earnings, analyst, insider,
                                             rs_data, divergences, info)

        return {
            "ticker": ticker.upper(),
            "info": info,
            "technicals": technicals,
            "sentiment": sentiment,
            "rating": rating,
            "rating_history": rating_history,
            "analyst": analyst,
            "insider": insider,
            "earnings": earnings,
            "relative_strength": rs_data,
            "divergences": divergences,
            "sentiment_technical_agreement": agreement,
            "buy_timing": buy_timing,
            "written_analysis": written,
            "df": df,
            "error": None,
        }
    except ValueError as e:
        return {"ticker": ticker.upper(), "error": str(e)}
    except Exception as e:
        return {"ticker": ticker.upper(), "error": f"Error analyzing {ticker}: {e}"}


# --- Position Management ---

def load_positions() -> list[dict]:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return []


def save_positions(positions: list[dict]):
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2, default=str))


def add_position(ticker, entry_price, shares, entry_date,
                 stop_loss=None, target_price=None, notes="",
                 entry_sentiment_score=None):
    positions = load_positions()
    pos = {
        "id": f"{ticker.upper()}_{entry_date}_{len(positions)}",
        "ticker": ticker.upper(),
        "entry_price": entry_price,
        "shares": shares,
        "entry_date": entry_date,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "notes": notes,
        "entry_sentiment_score": entry_sentiment_score,
    }
    positions.append(pos)
    save_positions(positions)
    return positions


def remove_position(position_id: str) -> list[dict]:
    positions = load_positions()
    positions = [p for p in positions if p["id"] != position_id]
    save_positions(positions)
    return positions


def update_position(position_id: str, **fields) -> list[dict]:
    positions = load_positions()
    for p in positions:
        if p["id"] == position_id:
            for key, val in fields.items():
                if key in ("stop_loss", "target_price", "notes"):
                    p[key] = val
            break
    save_positions(positions)
    return positions


def close_position_with_history(position_id: str, exit_price: float = None) -> list[dict]:
    positions = load_positions()
    closed = None
    remaining = []
    for p in positions:
        if p["id"] == position_id:
            closed = p
        else:
            remaining.append(p)
    save_positions(remaining)

    if closed and exit_price:
        history = load_trade_history()
        pnl_pct = ((exit_price - closed["entry_price"]) / closed["entry_price"]) * 100
        entry_dt = datetime.strptime(closed["entry_date"], "%Y-%m-%d").date()
        days_held = (date.today() - entry_dt).days
        record = {
            "id": closed["id"],
            "ticker": closed["ticker"],
            "entry_price": closed["entry_price"],
            "exit_price": exit_price,
            "shares": closed["shares"],
            "entry_date": closed["entry_date"],
            "exit_date": date.today().strftime("%Y-%m-%d"),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_total": round((exit_price - closed["entry_price"]) * closed["shares"], 2),
            "days_held": days_held,
            "notes": closed.get("notes", ""),
        }
        history.append(record)
        HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))
    return remaining


def load_trade_history() -> list[dict]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def compute_trade_statistics() -> dict:
    history = load_trade_history()
    if not history:
        return {"has_data": False, "total_trades": 0}
    wins = [t for t in history if t["pnl_pct"] > 0]
    losses = [t for t in history if t["pnl_pct"] <= 0]
    avg_gain = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(history) * 100
    profit_factor = (sum(t["pnl_total"] for t in wins) / abs(sum(t["pnl_total"] for t in losses))) if losses and sum(t["pnl_total"] for t in losses) != 0 else float("inf")
    total_pnl = sum(t["pnl_total"] for t in history)
    avg_hold = sum(t["days_held"] for t in history) / len(history)
    best = max(history, key=lambda t: t["pnl_pct"])
    worst = min(history, key=lambda t: t["pnl_pct"])
    return {
        "has_data": True,
        "total_trades": len(history),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_gain": round(avg_gain, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "total_pnl": round(total_pnl, 2),
        "avg_hold_days": round(avg_hold, 1),
        "best_trade": best,
        "worst_trade": worst,
        "history": history,
    }


def analyze_position(position, analysis, fx_rate=1.0):
    if analysis.get("error"):
        return {"action": "Unknown", "reasons": [analysis["error"]], "urgency": "low",
                "days_held": 0, "trading_days_held": 0, "pnl_pct": 0, "pnl_total": 0,
                "cost_basis": 0, "market_value": 0, "current_price": 0, "entry_price": 0}

    def cad(usd_val):
        return round(usd_val * fx_rate, 2)

    tech = analysis["technicals"]
    current_price = tech["current_price"]
    entry_price = position["entry_price"]
    shares = position["shares"]

    pnl_per_share = current_price - entry_price
    pnl_total = pnl_per_share * shares
    pnl_pct = (pnl_per_share / entry_price) * 100
    cost_basis = entry_price * shares
    market_value = current_price * shares

    entry_date = datetime.strptime(position["entry_date"], "%Y-%m-%d").date()
    days_held = (date.today() - entry_date).days
    trading_days_held = int(days_held * 5 / 7)

    reasons = []
    sell_pressure = 0
    hold_pressure = 0

    atr = tech.get("atr", 0)
    atr_pct = tech.get("atr_pct", 0)

    if trading_days_held > 15:
        sell_pressure += 25
        reasons.append(f"Held {days_held} days ({trading_days_held} trading days) — past the 3-week swing window")
    elif trading_days_held > 10:
        sell_pressure += 10
        reasons.append(f"Held {days_held} days — approaching end of 1-3 week swing window")
    else:
        hold_pressure += 10
        reasons.append(f"Held {days_held} days — still within swing timeframe")

    if atr > 0 and pnl_per_share > 0:
        atr_multiple = pnl_per_share / atr
        if atr_multiple >= 3:
            sell_pressure += 25
            reasons.append(f"Up {atr_multiple:.1f}x ATR — exceeded 3:1 risk/reward target")
        elif atr_multiple >= 2:
            sell_pressure += 10
            reasons.append(f"Up {atr_multiple:.1f}x ATR — solid profit, trail your stop to C${cad(current_price - 1.5 * atr)}")
        else:
            hold_pressure += 5
            reasons.append(f"Up {atr_multiple:.1f}x ATR — room to reach 3:1 target")
    elif pnl_pct >= 10:
        sell_pressure += 25
        reasons.append(f"Up {pnl_pct:.1f}% — strong profit, consider locking in gains")
    elif pnl_pct >= 5:
        sell_pressure += 10
        reasons.append(f"Up {pnl_pct:.1f}% — solid profit, trail your stop")
    elif pnl_pct > 0:
        hold_pressure += 5
        reasons.append(f"Up {pnl_pct:.1f}% — small gain, let it work")
    elif pnl_pct > -3:
        hold_pressure += 5
        reasons.append(f"Down {abs(pnl_pct):.1f}% — minor drawdown, still within normal range")
    elif pnl_pct > -7:
        sell_pressure += 15
        reasons.append(f"Down {abs(pnl_pct):.1f}% — notable loss, watch closely for trend break")
    else:
        sell_pressure += 30
        reasons.append(f"Down {abs(pnl_pct):.1f}% — significant loss, strongly consider cutting")

    if position.get("stop_loss"):
        if current_price <= position["stop_loss"]:
            sell_pressure += 40
            reasons.append(f"STOP LOSS HIT — price C${cad(current_price)} at or below stop C${cad(position['stop_loss'])}")
        elif current_price < position["stop_loss"] * 1.02:
            sell_pressure += 15
            reasons.append(f"Price approaching stop loss (C${cad(current_price)} vs stop C${cad(position['stop_loss'])})")

    if position.get("target_price"):
        if current_price >= position["target_price"]:
            sell_pressure += 30
            reasons.append(f"TARGET REACHED — price C${cad(current_price)} hit target C${cad(position['target_price'])}")
        elif current_price >= position["target_price"] * 0.98:
            sell_pressure += 10
            reasons.append(f"Approaching target (C${cad(current_price)} vs target C${cad(position['target_price'])})")

    earnings = analysis.get("earnings", {})
    if earnings.get("has_date") and earnings.get("in_swing_window"):
        days_to_earnings = earnings["days_until"]
        if days_to_earnings <= 5:
            sell_pressure += 20
            reasons.append(f"EARNINGS in {days_to_earnings} days ({earnings['date_str']}) — high volatility risk, consider closing or sizing down")
        elif days_to_earnings <= 14:
            sell_pressure += 8
            reasons.append(f"Earnings in {days_to_earnings} days ({earnings['date_str']}) — be aware of event risk within swing window")

    entry_sent = position.get("entry_sentiment_score")
    current_sent = analysis.get("sentiment", {}).get("sentiment_score", 50)
    if entry_sent is not None:
        sent_shift = current_sent - entry_sent
        if sent_shift < -20:
            sell_pressure += 10
            reasons.append(f"Sentiment deteriorated since entry (was {entry_sent:.0f}, now {current_sent:.0f}) — negative news shift")
        elif sent_shift > 20:
            hold_pressure += 10
            reasons.append(f"Sentiment improved since entry (was {entry_sent:.0f}, now {current_sent:.0f}) — positive news flow supporting position")

    rsi = tech["rsi"]
    if pnl_pct > 0:
        if rsi > 70:
            sell_pressure += 20
            reasons.append(f"RSI overbought at {rsi} while in profit — momentum may fade")
        elif rsi > 60:
            hold_pressure += 5
            reasons.append(f"RSI at {rsi} — momentum still healthy")
    else:
        if rsi < 30:
            hold_pressure += 15
            reasons.append(f"RSI oversold at {rsi} — may bounce, consider averaging down or holding")
        elif rsi < 40:
            hold_pressure += 5
            reasons.append(f"RSI at {rsi} — approaching oversold, bounce possible")

    if tech["macd_recent_crossover"] and tech["macd_crossover_direction"] == "bearish":
        sell_pressure += 15
        reasons.append("Fresh bearish MACD crossover — momentum turning against this position")
    elif tech["macd_recent_crossover"] and tech["macd_crossover_direction"] == "bullish":
        hold_pressure += 15
        reasons.append("Fresh bullish MACD crossover — momentum supporting this position")
    elif tech["macd_hist"] < 0 and tech["macd_hist"] < tech["macd_hist_prev"]:
        sell_pressure += 10
        reasons.append("MACD histogram negative and expanding — bearish momentum building")
    elif tech["macd_hist"] > 0 and tech["macd_hist"] > tech["macd_hist_prev"]:
        hold_pressure += 10
        reasons.append("MACD histogram positive and expanding — bullish momentum intact")

    if "bearish" in tech["trend"]:
        sell_pressure += 15
        reasons.append(f"Trend is {tech['trend']} — price below key moving averages")
    elif "bullish" in tech["trend"]:
        hold_pressure += 15
        reasons.append(f"Trend is {tech['trend']} — moving averages support the position")

    if pnl_pct < 0 and tech["support_levels"]:
        nearest_support = tech["support_levels"][0]
        dist_to_support = ((current_price - nearest_support) / current_price) * 100
        if dist_to_support < 1:
            hold_pressure += 10
            reasons.append(f"Price near support at C${cad(nearest_support)} — may hold here")
        elif current_price < nearest_support:
            sell_pressure += 15
            reasons.append(f"Price broke below nearest support C${cad(nearest_support)}")

    if pnl_pct > 0 and tech["resistance_levels"]:
        nearest_resistance = tech["resistance_levels"][0]
        dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
        if dist_to_resistance < 1:
            sell_pressure += 10
            reasons.append(f"Price at resistance C${cad(nearest_resistance)} — may stall here")

    net = sell_pressure - hold_pressure
    if net >= 50:
        action = "Sell Now"
        urgency = "high"
    elif net >= 30:
        action = "Consider Selling"
        urgency = "medium"
    elif net >= 10:
        action = "Tighten Stop"
        urgency = "low"
    elif net >= -10:
        action = "Hold"
        urgency = "low"
    elif net >= -30:
        action = "Hold — Looking Good"
        urgency = "low"
    else:
        action = "Strong Hold"
        urgency = "low"

    atr_stop = round(entry_price - 2 * atr, 2) if atr > 0 else None
    atr_target = round(entry_price + 3 * atr, 2) if atr > 0 else None
    atr_trail = round(current_price - 1.5 * atr, 2) if atr > 0 and pnl_pct > 0 else None

    return {
        "action": action,
        "urgency": urgency,
        "sell_pressure": sell_pressure,
        "hold_pressure": hold_pressure,
        "reasons": reasons,
        "pnl_per_share": round(pnl_per_share, 2),
        "pnl_total": round(pnl_total, 2),
        "pnl_pct": round(pnl_pct, 2),
        "cost_basis": round(cost_basis, 2),
        "market_value": round(market_value, 2),
        "days_held": days_held,
        "trading_days_held": trading_days_held,
        "current_price": current_price,
        "entry_price": entry_price,
        "atr_suggested_stop": atr_stop,
        "atr_suggested_target": atr_target,
        "atr_trailing_stop": atr_trail,
    }


# --- Watchlist Management ---

def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return []


def save_watchlist(tickers: list[str]):
    WATCHLIST_FILE.write_text(json.dumps(tickers, indent=2))


def add_to_watchlist(ticker: str) -> list[str]:
    wl = load_watchlist()
    t = ticker.upper().strip()
    if t and t not in wl:
        wl.append(t)
        save_watchlist(wl)
    return wl


def remove_from_watchlist(ticker: str) -> list[str]:
    wl = load_watchlist()
    wl = [t for t in wl if t != ticker.upper().strip()]
    save_watchlist(wl)
    return wl


def fetch_market_movers_news(tickers: list[str], min_compound: float = 0.4) -> list[dict]:
    sia = SentimentIntensityAnalyzer()
    big_stories = []
    seen_titles = set()
    for ticker in tickers:
        try:
            articles = fetch_news(ticker, count=5)
            for article in articles:
                title = article.get("title", "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                scores = sia.polarity_scores(title)
                compound = scores["compound"]
                if abs(compound) >= min_compound:
                    big_stories.append({
                        "ticker": ticker.upper(),
                        "title": title,
                        "publisher": article.get("publisher", "Unknown"),
                        "link": article.get("link", ""),
                        "compound": round(compound, 3),
                        "sentiment": "bullish" if compound > 0 else "bearish",
                    })
        except Exception:
            continue
    big_stories.sort(key=lambda x: abs(x["compound"]), reverse=True)
    return big_stories
