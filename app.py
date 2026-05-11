import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, datetime
from analysis import (
    analyze_ticker, load_positions, save_positions,
    add_position, remove_position, update_position,
    close_position_with_history, analyze_position,
    get_usd_cad_rate, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
    load_trade_history, compute_trade_statistics,
    explain_rating_change, fetch_market_movers_news,
)

st.set_page_config(page_title="TalicoTrading", page_icon="\U0001f4ca",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
section[data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid #1e2530; }
section[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; letter-spacing: -0.02em; }

button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 500 !important; padding: 10px 16px !important; }
button[data-baseweb="tab"][aria-selected="true"] { font-weight: 700 !important; }

[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }
[data-testid="stExpander"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px; margin-bottom: 8px;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; }

button[data-testid="stBaseButton-primary"] {
    border-radius: 8px !important; font-weight: 700 !important; letter-spacing: 0.02em;
    background-color: #1a73e8 !important; color: #ffffff !important; border: none !important;
    padding: 12px 24px !important; font-size: 0.95rem !important; cursor: pointer !important;
}
button[data-testid="stBaseButton-primary"]:hover { background-color: #1565c0 !important; }
button[data-testid="stBaseButton-primary"]:active { background-color: #0d47a1 !important; }

hr { border-color: #21262d !important; opacity: 0.5 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid #21262d !important; }
button[data-testid="stBaseButton-secondary"] { border-radius: 8px !important; }
[data-testid="stForm"] { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }

.rating-badge {
    display: inline-block; padding: 14px 20px; border-radius: 12px;
    text-align: center; font-weight: 700; line-height: 1.3;
}
.rating-badge .label { font-size: 1.3em; color: #000; }
.rating-badge .score { font-size: 1.1em; color: #000; }

.section-header {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #8b949e; font-weight: 600; margin-bottom: 8px; margin-top: 16px;
}

.timing-badge {
    display: inline-block; padding: 10px 18px; border-radius: 10px;
    text-align: center; font-weight: 700; font-size: 1.1em; line-height: 1.3;
}
.timing-badge .sub { font-size: 0.7em; font-weight: 400; opacity: 0.8; }

.news-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
}

.signal-row {
    padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

RATING_COLORS = {
    "Strong Buy": "#00c853", "Buy": "#69f0ae", "Neutral": "#ffd600",
    "Sell": "#ff5252", "Strong Sell": "#d50000",
}
TIMING_COLORS = {
    "Buy Now": "#00c853",
    "Buy — Pullback Entry": "#69f0ae",
    "Watch for Entry": "#ffd600",
    "Risky Entry": "#ff9100",
    "Overextended": "#ff5252",
    "Wait for Pullback": "#ffd600",
    "Avoid for Now": "#d50000",
}
ACTION_COLORS = {
    "Sell Now": "#d50000", "Consider Selling": "#ff5252",
    "Tighten Stop": "#ffd600", "Hold": "#ffd600",
    "Hold — Looking Good": "#69f0ae", "Strong Hold": "#00c853", "Unknown": "#757575",
}
SCREENER_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "CRM",
    "AVGO", "ADBE", "ORCL", "INTC", "QCOM", "MU", "SHOP", "SQ", "COIN", "PLTR",
    "JPM", "BAC", "GS", "V", "MA", "PYPL",
    "XOM", "CVX", "OXY",
    "UNH", "JNJ", "PFE", "MRNA", "LLY",
    "DIS", "NKE", "SBUX", "HD", "WMT", "COST",
    "BA", "CAT", "DE",
    "SPY", "QQQ", "IWM",
]


@st.cache_data(ttl=300)
def cached_analyze(ticker: str, period: str) -> dict:
    return analyze_ticker(ticker, period)

@st.cache_data(ttl=300)
def cached_fx_rate() -> float:
    return get_usd_cad_rate()

@st.cache_data(ttl=600)
def cached_market_movers() -> list:
    return fetch_market_movers_news(SCREENER_TICKERS, min_compound=0.4)

def usd_to_cad(usd_val):
    return round(usd_val * cached_fx_rate(), 2)

def trend_arrow(history, current_score):
    if not history:
        return ""
    prev = history[-1]["score"]
    diff = current_score - prev
    if diff >= 5:
        return " ↑"
    elif diff <= -5:
        return " ↓"
    return " →"

def score_trail(history, current_score):
    parts = [str(int(h["score"])) for h in history]
    parts.append(str(int(current_score)))
    return " → ".join(parts)


_chart_counter = [0]

def _next_key(prefix="chart"):
    _chart_counter[0] += 1
    return f"{prefix}_{_chart_counter[0]}"


# ── Indicator explanation helper ──

def explain_indicator(name, value, technicals, for_buying=True):
    t = technicals
    if name == "RSI":
        rsi = t["rsi"]
        direction = "rising" if t.get("rsi_prev", rsi) < rsi else "falling"
        if rsi < 30:
            return ("Oversold — historically strong reversal zone. Buyers often step in here.",
                    "Supports buying", "strong", "#00c853")
        elif rsi < 40:
            return (f"Approaching oversold ({direction}). Room for upside if trend holds.",
                    "Supports buying", "moderate", "#69f0ae")
        elif rsi < 55:
            return (f"Neutral territory ({direction}). No strong directional signal from RSI alone.",
                    "Neutral", "weak", "#ffd600")
        elif rsi < 70:
            return (f"Elevated ({direction}). Momentum is up but getting stretched.",
                    "Warns against buying", "moderate", "#ff9100")
        else:
            return (f"Overbought ({direction}). High risk of pullback from these levels.",
                    "Warns against buying", "strong", "#ff5252")

    elif name == "MACD":
        if t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bullish":
            return ("Fresh bullish crossover — momentum just shifted positive. Strong entry signal.",
                    "Supports buying", "strong", "#00c853")
        elif t["macd_recent_crossover"] and t["macd_crossover_direction"] == "bearish":
            return ("Fresh bearish crossover — momentum turning negative. Avoid new entries.",
                    "Warns against buying", "strong", "#ff5252")
        elif t["macd_hist"] > 0 and t["macd_hist"] > t["macd_hist_prev"]:
            return ("Histogram positive and expanding — bullish momentum building.",
                    "Supports buying", "moderate", "#69f0ae")
        elif t["macd_hist"] > 0:
            return ("Histogram positive but contracting — bullish momentum fading.",
                    "Neutral", "weak", "#ffd600")
        elif t["macd_hist"] < 0 and t["macd_hist"] > t["macd_hist_prev"]:
            return ("Histogram negative but converging — selling pressure easing.",
                    "Neutral", "weak", "#ffd600")
        else:
            return ("Histogram negative and expanding — bearish momentum increasing.",
                    "Warns against buying", "moderate", "#ff5252")

    elif name == "Trend":
        trend = t["trend"]
        if "strong bullish" in trend:
            return ("All MAs aligned bullish — strong uptrend. Price above SMA20 > SMA50 > SMA200.",
                    "Supports buying", "strong", "#00c853")
        elif "bullish" in trend:
            return ("Price above short-term MAs — uptrend in place.",
                    "Supports buying", "moderate", "#69f0ae")
        elif "strong bearish" in trend:
            return ("All MAs aligned bearish — strong downtrend. Avoid longs.",
                    "Warns against buying", "strong", "#ff5252")
        elif "bearish" in trend:
            return ("Price below short-term MAs — downtrend bias.",
                    "Warns against buying", "moderate", "#ff5252")
        else:
            return ("Mixed MA alignment — no clear directional bias.",
                    "Neutral", "weak", "#ffd600")

    elif name == "Volume":
        vr = t["vol_ratio"]
        if vr > 1.3 and t["price_direction"] == "up":
            return (f"Volume {vr:.1f}x average on up day — buyers stepping in with conviction.",
                    "Supports buying", "strong" if vr > 1.5 else "moderate", "#00c853")
        elif vr > 1.3 and t["price_direction"] == "down":
            return (f"Volume {vr:.1f}x average on down day — selling pressure is real.",
                    "Warns against buying", "strong" if vr > 1.5 else "moderate", "#ff5252")
        else:
            return (f"Volume near average ({vr:.1f}x) — no strong conviction from volume.",
                    "Neutral", "weak", "#ffd600")

    elif name == "Volatility":
        atr_pct = t["atr_pct"]
        if atr_pct > 4:
            return (f"High volatility (ATR {atr_pct:.1f}% of price). Wider stops needed, bigger swings.",
                    "Warns against buying" if atr_pct > 6 else "Neutral", "moderate", "#ff9100")
        elif atr_pct > 2:
            return (f"Moderate volatility (ATR {atr_pct:.1f}%). Normal for swing trading.",
                    "Neutral", "weak", "#ffd600")
        else:
            return (f"Low volatility (ATR {atr_pct:.1f}%). Tight range — breakout or breakdown may be coming.",
                    "Neutral", "weak", "#ffd600")

    elif name == "Bollinger":
        bp = t["bb_position"]
        if bp < 0.15:
            return ("Price near lower band — potential bounce zone. Often marks short-term bottoms.",
                    "Supports buying", "moderate", "#69f0ae")
        elif bp > 0.85:
            return ("Price near upper band — overextended. Pullback likely before next move up.",
                    "Warns against buying", "moderate", "#ff5252")
        else:
            return (f"Price in middle of bands ({bp:.0%}). No strong mean-reversion signal.",
                    "Neutral", "weak", "#ffd600")

    elif name == "Support/Resistance":
        parts = []
        verdict = "Neutral"
        color = "#ffd600"
        if t["support_levels"]:
            ns = t["support_levels"][0]
            sd = ((t["current_price"] - ns) / t["current_price"]) * 100
            parts.append(f"Nearest support: ${ns} ({sd:.1f}% below)")
            if sd < 2:
                verdict = "Supports buying"
                color = "#00c853"
                parts.append("Price very close to support — tight stop, good risk/reward.")
        if t["resistance_levels"]:
            nr = t["resistance_levels"][0]
            rd = ((nr - t["current_price"]) / t["current_price"]) * 100
            parts.append(f"Nearest resistance: ${nr} ({rd:.1f}% above)")
            if rd < 1.5:
                verdict = "Warns against buying"
                color = "#ff9100"
                parts.append("Pressing against resistance — limited upside until breakout.")
        if not parts:
            parts.append("No clear support/resistance levels detected nearby.")
        return (" ".join(parts), verdict, "moderate" if verdict != "Neutral" else "weak", color)

    elif name == "Recent Action":
        r1 = t.get("ret_1d", 0)
        r5 = t.get("ret_5d", 0)
        cu = t.get("consec_up", 0)
        cd = t.get("consec_down", 0)
        parts = [f"1-day: {r1:+.1f}%, 5-day: {r5:+.1f}%."]
        if cu >= 4:
            parts.append(f"{cu} consecutive up days — short-term exhaustion risk.")
            return (" ".join(parts), "Warns against buying", "moderate", "#ff9100")
        elif cd >= 4 and t["rsi"] < 40:
            parts.append(f"{cd} consecutive down days with RSI {t['rsi']:.0f} — washout may be ending.")
            return (" ".join(parts), "Supports buying", "moderate", "#69f0ae")
        elif r5 < -5:
            parts.append("Sharp 5-day decline — watch for reversal or continued selling.")
            return (" ".join(parts), "Neutral", "moderate", "#ffd600")
        elif r5 > 5:
            parts.append("Strong 5-day rally — momentum is up but entry risk elevated.")
            return (" ".join(parts), "Neutral", "moderate", "#ffd600")
        return (" ".join(parts), "Neutral", "weak", "#ffd600")

    return ("No data available.", "Neutral", "weak", "#757575")


# ── Render helpers ──

def render_buy_timing_badge(result):
    bt = result.get("buy_timing", {})
    timing = bt.get("timing", "Watch for Entry")
    confidence = bt.get("confidence", "low")
    tc = TIMING_COLORS.get(timing, "#757575")
    st.markdown(
        f"<div class='timing-badge' style='background:{tc};color:#000'>"
        f"{timing}<br><span class='sub'>Confidence: {confidence.title()}</span></div>",
        unsafe_allow_html=True)


def render_written_analysis(result):
    wa = result.get("written_analysis", {})
    bt = result.get("buy_timing", {})
    if not wa:
        return

    st.markdown(wa.get("summary", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">\U0001f7e2 Bullish Signals</p>', unsafe_allow_html=True)
        for sig in wa.get("bull_signals", []):
            st.markdown(f"<div class='signal-row' style='background:rgba(0,200,83,0.08);border-left:3px solid #00c853'>"
                        f"{sig}</div>", unsafe_allow_html=True)
        if not wa.get("bull_signals"):
            st.caption("No strong bullish signals right now")

    with col2:
        st.markdown('<p class="section-header">\U0001f534 Bearish Signals</p>', unsafe_allow_html=True)
        for sig in wa.get("bear_signals", []):
            st.markdown(f"<div class='signal-row' style='background:rgba(255,82,82,0.08);border-left:3px solid #ff5252'>"
                        f"{sig}</div>", unsafe_allow_html=True)
        if not wa.get("bear_signals"):
            st.caption("No strong bearish signals right now")

    if wa.get("risks"):
        st.markdown('<p class="section-header">⚠️ What Could Go Wrong</p>', unsafe_allow_html=True)
        for risk in wa["risks"]:
            st.markdown(f"- {risk}")

    if bt.get("better_entry"):
        st.markdown('<p class="section-header">\U0001f4a1 What Would Make a Better Entry</p>', unsafe_allow_html=True)
        for tip in bt["better_entry"]:
            st.markdown(f"- {tip}")

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown('<p class="section-header">Short-Term Outlook (1-2 weeks)</p>', unsafe_allow_html=True)
        st.markdown(wa.get("short_term_outlook", ""))
    with oc2:
        st.markdown('<p class="section-header">Medium-Term Outlook (2-6 weeks)</p>', unsafe_allow_html=True)
        st.markdown(wa.get("medium_term_outlook", ""))
    with oc3:
        st.markdown('<p class="section-header">Confidence Level</p>', unsafe_allow_html=True)
        st.markdown(wa.get("confidence", ""))


def render_indicators_explained(result):
    tech = result["technicals"]
    indicators = ["RSI", "MACD", "Trend", "Volume", "Volatility", "Bollinger", "Support/Resistance", "Recent Action"]
    values = [
        f"{tech['rsi']:.1f}",
        f"{tech['macd_hist']:.4f}",
        tech["trend"].title(),
        f"{tech['vol_ratio']:.1f}x avg",
        f"{tech['atr_pct']:.1f}%",
        f"{tech['bb_position']:.0%}",
        f"S: {len(tech['support_levels'])} / R: {len(tech['resistance_levels'])}",
        f"{tech.get('ret_1d', 0):+.1f}% today",
    ]
    for name, val in zip(indicators, values):
        explanation, verdict, strength, color = explain_indicator(name, val, tech)
        strength_bar = "●●●" if strength == "strong" else "●●○" if strength == "moderate" else "●○○"
        v_color = "#00c853" if "Supports" in verdict else "#ff5252" if "Warns" in verdict else "#ffd600"
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #21262d;border-left:3px solid {color};"
            f"border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-weight:700'>{name}</span>"
            f"<span style='font-size:0.85em'>{val} &nbsp;"
            f"<span style='color:{v_color}'>{verdict}</span> &nbsp;"
            f"<span style='opacity:0.5'>{strength_bar}</span></span></div>"
            f"<div style='font-size:0.85em;opacity:0.8;margin-top:4px'>{explanation}</div></div>",
            unsafe_allow_html=True)


def render_news_sentiment(result):
    sent = result["sentiment"]
    if not sent.get("has_news"):
        st.info("No recent news found. Sentiment defaults to neutral.")
        return

    overall = sent["overall_sentiment"]
    ov_color = "#00c853" if overall == "bullish" else "#ff5252" if overall == "bearish" else "#ffd600"
    strength = sent.get("strength", "none")

    nc1, nc2 = st.columns([1, 3])
    with nc1:
        st.markdown(
            f"<div style='background:{ov_color};color:#000;padding:20px;border-radius:12px;"
            f"text-align:center'>"
            f"<div style='font-size:1.4em;font-weight:800'>{overall.title()}</div>"
            f"<div style='font-size:0.85em;opacity:0.7'>Sentiment</div>"
            f"<div style='margin-top:6px;font-size:0.9em'>{strength.title()} signal</div>"
            f"<div style='margin-top:4px;font-size:0.85em'>"
            f"\U0001f7e2 {sent['bullish_count']} &nbsp; ⚪ {sent['neutral_count']} &nbsp; "
            f"\U0001f534 {sent['bearish_count']}</div></div>",
            unsafe_allow_html=True)

    with nc2:
        total = sent["bullish_count"] + sent["neutral_count"] + sent["bearish_count"]
        if total > 0:
            bull_pct = sent["bullish_count"] / total * 100
            neut_pct = sent["neutral_count"] / total * 100
            bear_pct = sent["bearish_count"] / total * 100
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(y=["Sentiment"], x=[bull_pct], orientation="h",
                                      marker_color="#00c853", name="Bullish", text=f"{bull_pct:.0f}%",
                                      textposition="inside"))
            fig_bar.add_trace(go.Bar(y=["Sentiment"], x=[neut_pct], orientation="h",
                                      marker_color="#757575", name="Neutral", text=f"{neut_pct:.0f}%",
                                      textposition="inside"))
            fig_bar.add_trace(go.Bar(y=["Sentiment"], x=[bear_pct], orientation="h",
                                      marker_color="#ff5252", name="Bearish", text=f"{bear_pct:.0f}%",
                                      textposition="inside"))
            fig_bar.update_layout(barmode="stack", template="plotly_dark", height=80,
                                  margin=dict(l=10, r=10, t=5, b=5), showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.1),
                                  xaxis=dict(showticklabels=False, range=[0, 100]),
                                  yaxis=dict(showticklabels=False))
            st.plotly_chart(fig_bar, use_container_width=True, key=_next_key("sent_bar"))

        st.markdown(f"> {sent['significance']}")

    st.markdown('<p class="section-header">Recent Headlines</p>', unsafe_allow_html=True)
    for article in sent["articles"][:8]:
        s = article["compound_score"]
        if s > 0.2:
            border_col = "#00c853"
            icon = "\U0001f7e2"
        elif s < -0.2:
            border_col = "#ff5252"
            icon = "\U0001f534"
        else:
            border_col = "#757575"
            icon = "⚪"
        sc_color = "#00c853" if s > 0 else "#ff5252" if s < 0 else "#757575"
        st.markdown(
            f"<div class='news-card' style='border-left:3px solid {border_col}'>"
            f"{icon} <b>{article['title']}</b><br>"
            f"<span style='opacity:0.5;font-size:0.85em'>{article['publisher']}</span> &nbsp; "
            f"<span style='color:{sc_color};font-size:0.85em;font-weight:600'>Score: {s:.2f}</span>"
            f"</div>", unsafe_allow_html=True)


def render_rating_history(result):
    history = result.get("rating_history", [])
    if not history:
        st.caption("Not enough data for rating history.")
        return

    chart_key = _next_key("rh")
    current = result["rating"]
    all_days = list(history) + [{
        "date": "Today", "score": current["combined_score"],
        "rating": current["rating"], "component_scores": current["component_scores"],
        "change_explanation": None,
    }]
    if len(history) > 0:
        today_prev = history[-1]
        all_days[-1]["change_explanation"] = explain_rating_change(today_prev, {
            "score": current["combined_score"], "rating": current["rating"],
            "component_scores": current["component_scores"],
        })

    all_days_reversed = list(reversed(all_days))

    st.markdown('<p class="section-header">Rating History — Last 5 Days + Today</p>', unsafe_allow_html=True)
    cols = st.columns(len(all_days_reversed))
    for i, day in enumerate(all_days_reversed):
        rc = RATING_COLORS.get(day["rating"], "#757575")
        is_today = day["date"] == "Today"
        border = "3px solid rgba(255,255,255,0.8)" if is_today else "1px solid #333"
        with cols[i]:
            st.markdown(
                f"<div style='background:{rc};color:#000;padding:8px 4px;border-radius:10px;"
                f"text-align:center;font-size:0.8em;border:{border}'>"
                f"<div style='font-weight:700;font-size:0.85em'>{day['date'] if not is_today else 'Today'}</div>"
                f"<div style='font-size:1.3em;font-weight:800'>{day['score']:.0f}</div>"
                f"<div style='font-size:0.7em;opacity:0.7'>{day['rating']}</div></div>",
                unsafe_allow_html=True)

    with st.expander("Day-by-Day Breakdown", expanded=False):
        for day in all_days:
            expl = day.get("change_explanation")
            if expl is None:
                continue
            if isinstance(expl, str):
                st.markdown(f"**{day['date']}:** {expl}")
                continue
            summary = expl.get("summary", "")
            details = expl.get("details", [])
            st.markdown(f"**{day['date']}** — {summary}")
            for d in details:
                color = "#00c853" if d["delta"] > 0 else "#ff5252"
                icon = "▲" if d["delta"] > 0 else "▼"
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{color}'>{icon}</span> {d['text']}",
                            unsafe_allow_html=True)

    if len(all_days) >= 2:
        scores = [d["score"] for d in all_days]
        dates = [d["date"] for d in all_days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers+text",
                                  text=[f"{s:.0f}" for s in scores], textposition="top center",
                                  line=dict(color="#42a5f5", width=3), marker=dict(size=10)))
        fig.add_hline(y=80, line_dash="dash", line_color="#00c853", opacity=0.3)
        fig.add_hline(y=65, line_dash="dash", line_color="#69f0ae", opacity=0.3)
        fig.add_hline(y=45, line_dash="dash", line_color="#ffd600", opacity=0.2)
        fig.update_layout(template="plotly_dark", height=200,
                          yaxis=dict(range=[0, 105], title="Score"),
                          margin=dict(l=50, r=50, t=10, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_catalysts(result):
    info = result["info"]
    analyst = result.get("analyst", {})
    insider = result.get("insider", {})
    earnings = result.get("earnings", {})
    rs = result.get("relative_strength", {})
    divergences = result.get("divergences", [])
    agreement = result.get("sentiment_technical_agreement", "neutral")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="section-header">Analyst Consensus</p>', unsafe_allow_html=True)
        if info.get("targetMeanPrice"):
            price = result["technicals"]["current_price"]
            target = info["targetMeanPrice"]
            upside = ((target - price) / price) * 100
            color = "#00c853" if upside > 0 else "#ff5252"
            st.markdown(f"Mean Target: **C${usd_to_cad(target)}** "
                        f"(<span style='color:{color}'>{upside:+.1f}%</span>)", unsafe_allow_html=True)
            st.caption(f"Range: C${usd_to_cad(info.get('targetLowPrice', 0))} — C${usd_to_cad(info.get('targetHighPrice', 0))}")
            st.caption(f"Analysts: {info.get('numberOfAnalystOpinions', 'N/A')} | {info.get('recommendationKey', 'N/A')}")
        else:
            st.caption("No analyst target data")
        if analyst.get("has_data"):
            st.markdown(f"Votes: **{analyst['strong_buy']+analyst['buy']}** Buy / "
                        f"**{analyst['hold']}** Hold / **{analyst['sell']+analyst['strong_sell']}** Sell")

    with c2:
        st.markdown('<p class="section-header">Earnings</p>', unsafe_allow_html=True)
        if earnings.get("has_date"):
            days = earnings["days_until"]
            icon = "\U0001f6a8" if days <= 5 else "⚠️" if days <= 14 else "\U0001f4c5"
            st.markdown(f"{icon} **{earnings['date_str']}** ({days} days)")
            if earnings.get("eps_estimate") is not None:
                st.markdown(f"EPS Est: **${earnings['eps_estimate']:.2f}**")
            if earnings["in_swing_window"]:
                st.warning("Earnings within swing window")
        else:
            st.caption("No upcoming earnings date")
        if earnings.get("history"):
            avg_s = earnings.get("avg_surprise_pct", 0)
            avg_c = "#00c853" if avg_s > 0 else "#ff5252"
            st.markdown(f"Track: **{earnings['beat_count']}** beat / **{earnings['miss_count']}** miss "
                        f"(avg <span style='color:{avg_c}'>{avg_s:+.1f}%</span>)", unsafe_allow_html=True)

    with c3:
        st.markdown('<p class="section-header">Relative Strength vs SPY</p>', unsafe_allow_html=True)
        if rs.get("has_data"):
            color = "#00c853" if rs["rs_ratio"] > 2 else "#ff5252" if rs["rs_ratio"] < -2 else "#ffd600"
            st.markdown(f"<span style='color:{color};font-weight:700'>{rs['rs_ratio']:+.1f}%</span> "
                        f"({rs['label']})", unsafe_allow_html=True)
            st.caption(f"Stock: {rs['stock_return']:+.1f}% vs SPY: {rs['spy_return']:+.1f}%")
        else:
            st.caption("Unavailable")

        if info.get("beta"):
            st.markdown(f"Beta: **{info['beta']:.2f}**")

        st.markdown('<p class="section-header">Short Interest</p>', unsafe_allow_html=True)
        if info.get("shortPercentOfFloat"):
            spf = info["shortPercentOfFloat"] * 100
            color = "#ff5252" if spf > 10 else "#ffd600" if spf > 5 else "#00c853"
            st.markdown(f"<span style='color:{color};font-weight:700'>{spf:.1f}%</span> of float", unsafe_allow_html=True)
        else:
            st.caption("No short interest data")

    if info.get("fiftyTwoWeekHigh") and info.get("fiftyTwoWeekLow"):
        price = result["technicals"]["current_price"]
        high = info["fiftyTwoWeekHigh"]
        low = info["fiftyTwoWeekLow"]
        pos = (price - low) / (high - low) if high != low else 0.5
        st.markdown('<p class="section-header">52-Week Range</p>', unsafe_allow_html=True)
        fig_range = go.Figure(go.Bar(x=[pos * 100], y=[""], orientation="h",
                                      marker_color="#42a5f5", width=0.3))
        fig_range.update_layout(template="plotly_dark", height=50,
                                xaxis=dict(range=[0, 100], tickvals=[0, 50, 100],
                                           ticktext=[f"C${usd_to_cad(low)}", "Mid", f"C${usd_to_cad(high)}"]),
                                margin=dict(l=10, r=10, t=5, b=25), showlegend=False)
        st.plotly_chart(fig_range, use_container_width=True, key=_next_key("range"))

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<p class="section-header">Insider Activity (90 days)</p>', unsafe_allow_html=True)
        if insider.get("has_data"):
            color = "#00c853" if insider["sentiment"] == "bullish" else "#ff5252" if insider["sentiment"] == "bearish" else "#ffd600"
            st.markdown(f"<span style='color:{color};font-weight:700'>{insider['sentiment'].title()}</span> — "
                        f"{insider['buy_count']} buys / {insider['sell_count']} sells "
                        f"(net ${insider['net_value']:,.0f})", unsafe_allow_html=True)
        else:
            st.caption("No insider data")

    with ic2:
        st.markdown('<p class="section-header">Sentiment + Technical Agreement</p>', unsafe_allow_html=True)
        if agreement == "aligned":
            st.markdown("\U0001f7e2 **Aligned** — Higher conviction setup")
        elif agreement == "divergent":
            st.markdown("\U0001f534 **Divergent** — Conflicting signals, use caution")
        else:
            st.markdown("⚪ **Neutral** — No strong agreement or conflict")

    if analyst.get("upgrades_downgrades"):
        st.markdown('<p class="section-header">Recent Analyst Actions</p>', unsafe_allow_html=True)
        ud_data = [{"Date": ud["date"], "Firm": ud["firm"], "Action": ud["action"],
                     "From": ud["from_grade"], "To": ud["to_grade"]}
                    for ud in analyst["upgrades_downgrades"][:5]]
        st.dataframe(pd.DataFrame(ud_data), hide_index=True, use_container_width=True)

    if divergences:
        st.markdown('<p class="section-header">Divergence Alerts</p>', unsafe_allow_html=True)
        for d in divergences:
            icon = "\U0001f7e2" if d["type"] == "bullish" else "\U0001f534"
            st.markdown(f"{icon} **{d['indicator']} {d['type'].title()}:** {d['description']}")


def render_analysis(result):
    if result.get("error"):
        st.error(result["error"])
        return

    info = result["info"]
    tech = result["technicals"]
    rating = result["rating"]
    df = result["df"]
    fx = cached_fx_rate()
    history = result.get("rating_history", [])
    bt = result.get("buy_timing", {})

    df_cad = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df_cad[col] = df_cad[col] * fx

    price_color = "green" if tech["price_change"] >= 0 else "red"
    rating_color = RATING_COLORS.get(rating["rating"], "#ffffff")
    timing_color = TIMING_COLORS.get(bt.get("timing", ""), "#757575")
    price_cad = usd_to_cad(tech["current_price"])
    arrow = trend_arrow(history, rating["combined_score"])

    st.markdown(f"### {result['ticker']} — {info['shortName']}")

    # Top row: Price + Buy Timing + Rating
    hc1, hc2, hc3 = st.columns([3, 1.5, 1.5])
    with hc1:
        pct_prefix = "+" if tech['price_change_pct'] >= 0 else ""
        r1 = tech.get("ret_1d", 0)
        r5 = tech.get("ret_5d", 0)
        st.markdown(
            f"<div style='line-height:1.6'>"
            f"<span style='font-size:1.6em;font-weight:700'>C${price_cad}</span> &nbsp;"
            f"<span style='color:{price_color};font-size:1em;font-weight:600'>"
            f"{pct_prefix}{tech['price_change_pct']}%</span>"
            f" &nbsp;<span style='opacity:0.5'>|</span>&nbsp; "
            f"<span style='opacity:0.6;font-size:0.9em'>5d: {r5:+.1f}%</span>"
            f" &nbsp;<span style='opacity:0.5'>|</span>&nbsp; "
            f"<span style='opacity:0.6;font-size:0.9em'>{info['sector']}</span>"
            f"</div>", unsafe_allow_html=True)
    with hc2:
        render_buy_timing_badge(result)
    with hc3:
        st.markdown(
            f"<div class='rating-badge' style='background:{rating_color}'>"
            f"<div class='label'>{rating['rating']}{arrow}</div>"
            f"<div class='score'>{rating['combined_score']:.0f}/100</div></div>",
            unsafe_allow_html=True)

    # Quick metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RSI", f"{tech['rsi']:.0f}",
              "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
    m2.metric("Trend", tech["trend"].title())
    m3.metric("Vol", f"{tech['vol_ratio']:.1f}x",
              "Above avg" if tech["vol_ratio"] > 1.2 else "Below avg" if tech["vol_ratio"] < 0.8 else "Avg")
    m4.metric("MACD", "Bullish" if tech["macd_hist"] > 0 else "Bearish",
              "Crossover!" if tech["macd_recent_crossover"] else
              "Expanding" if abs(tech["macd_hist"]) > abs(tech["macd_hist_prev"]) else "Contracting")
    m5.metric("ATR", f"{tech['atr_pct']:.1f}%", "High vol" if tech["atr_pct"] > 4 else "Normal")
    agr = result.get("sentiment_technical_agreement", "neutral")
    m6.metric("Agreement", agr.title(),
              "High conviction" if agr == "aligned" else "Conflicting" if agr == "divergent" else "")

    # Main tabs
    tab_verdict, tab_chart, tab_indicators, tab_news, tab_rating, tab_catalysts = st.tabs(
        ["📝 Verdict", "\U0001f4c8 Chart", "\U0001f527 Indicators", "\U0001f4f0 News", "⭐ Rating", "\U0001f680 Catalysts"])

    with tab_verdict:
        render_written_analysis(result)

    with tab_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df_cad.index, open=df_cad["Open"], high=df_cad["High"],
                                      low=df_cad["Low"], close=df_cad["Close"], name="Price (C$)"), row=1, col=1)
        if tech["sma_20"]:
            fig.add_trace(go.Scatter(x=df_cad.index, y=pd.Series(df_cad["Close"]).rolling(20).mean(),
                                     name="SMA 20", line=dict(color="orange", width=1)), row=1, col=1)
        if tech["sma_50"]:
            fig.add_trace(go.Scatter(x=df_cad.index, y=pd.Series(df_cad["Close"]).rolling(50).mean(),
                                     name="SMA 50", line=dict(color="cyan", width=1)), row=1, col=1)
        if tech["sma_200"]:
            fig.add_trace(go.Scatter(x=df_cad.index, y=pd.Series(df_cad["Close"]).rolling(200).mean(),
                                     name="SMA 200", line=dict(color="purple", width=1)), row=1, col=1)
        bb_upper = pd.Series(df_cad["Close"]).rolling(20).mean() + 2 * pd.Series(df_cad["Close"]).rolling(20).std()
        bb_lower = pd.Series(df_cad["Close"]).rolling(20).mean() - 2 * pd.Series(df_cad["Close"]).rolling(20).std()
        fig.add_trace(go.Scatter(x=df_cad.index, y=bb_upper, name="BB Upper",
                                 line=dict(color="rgba(150,150,150,0.3)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_cad.index, y=bb_lower, name="BB Lower",
                                 line=dict(color="rgba(150,150,150,0.3)", width=1),
                                 fill="tonexty", fillcolor="rgba(150,150,150,0.07)"), row=1, col=1)
        for level in tech["support_levels"]:
            fig.add_hline(y=usd_to_cad(level), line_dash="dash", line_color="green", line_width=1,
                          annotation_text=f"S {usd_to_cad(level)}", row=1, col=1)
        for level in tech["resistance_levels"]:
            fig.add_hline(y=usd_to_cad(level), line_dash="dash", line_color="red", line_width=1,
                          annotation_text=f"R {usd_to_cad(level)}", row=1, col=1)
        colors = ["green" if c >= o else "red" for c, o in zip(df_cad["Close"], df_cad["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=colors, opacity=0.6), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False,
                          margin=dict(l=50, r=50, t=30, b=30),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Price (C$)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True, key=_next_key("candle"))

    with tab_indicators:
        render_indicators_explained(result)

        st.markdown('<p class="section-header">Moving Averages</p>', unsafe_allow_html=True)
        ma_data = []
        for name, val in [("SMA 20", tech["sma_20"]), ("SMA 50", tech["sma_50"]),
                          ("SMA 200", tech["sma_200"]), ("EMA 12", tech["ema_12"]), ("EMA 26", tech["ema_26"])]:
            if val:
                dist = round(((tech["current_price"] - val) / val) * 100, 2)
                pos = "Above" if tech["current_price"] > val else "Below"
                ma_data.append({"MA": name, "Value (C$)": usd_to_cad(val), "Position": pos, "Distance": f"{dist}%"})
        if ma_data:
            st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Support Levels</p>', unsafe_allow_html=True)
            for lvl in tech["support_levels"]:
                dist = round(((tech["current_price"] - lvl) / lvl) * 100, 2)
                st.write(f"**C${usd_to_cad(lvl)}** — {dist}% below")
            if not tech["support_levels"]:
                st.caption("No clear support levels detected")
        with c2:
            st.markdown('<p class="section-header">Resistance Levels</p>', unsafe_allow_html=True)
            for lvl in tech["resistance_levels"]:
                dist = round(((lvl - tech["current_price"]) / tech["current_price"]) * 100, 2)
                st.write(f"**C${usd_to_cad(lvl)}** — {dist}% above")
            if not tech["resistance_levels"]:
                st.caption("No clear resistance levels detected")

    with tab_news:
        render_news_sentiment(result)

    with tab_rating:
        st.markdown(
            f"<div style='background:{rating_color};color:#000;padding:20px;border-radius:14px;"
            f"text-align:center;margin-bottom:16px'>"
            f"<div style='font-size:1.8em;font-weight:800'>{rating['rating']}{arrow}</div>"
            f"<div style='font-size:1.1em;font-weight:600;opacity:0.8'>"
            f"{rating['combined_score']:.0f} / 100</div></div>",
            unsafe_allow_html=True)

        render_rating_history(result)

        st.markdown('<p class="section-header">Component Breakdown</p>', unsafe_allow_html=True)
        comp_names = []
        comp_scores = []
        comp_colors = []
        for name, data in rating["component_scores"].items():
            comp_names.append(name.replace("_", " ").title())
            comp_scores.append(data["score"])
            comp_colors.append("#00c853" if data["score"] >= 60 else "#ff5252" if data["score"] <= 40 else "#ffd600")
        fig_bar = go.Figure(go.Bar(x=comp_scores, y=comp_names, orientation="h",
                                    marker_color=comp_colors,
                                    text=[f"{s:.0f}" for s in comp_scores], textposition="outside"))
        fig_bar.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.5)
        fig_bar.update_layout(template="plotly_dark", height=320,
                              xaxis=dict(range=[0, 105], title="Score"),
                              margin=dict(l=140, r=50, t=20, b=40))
        st.plotly_chart(fig_bar, use_container_width=True, key=_next_key("bar"))

        with st.expander("Component Reasoning", expanded=False):
            for name, data in rating["component_scores"].items():
                weight_pct = int(data["weight"] * 100)
                sc = "#00c853" if data["score"] >= 60 else "#ff5252" if data["score"] <= 40 else "#ffd600"
                st.markdown(
                    f"<span style='color:{sc};font-weight:700'>{name.replace('_', ' ').title()}</span> "
                    f"<span style='opacity:0.5'>({weight_pct}%)</span> — {data['reasoning']}",
                    unsafe_allow_html=True)

        st.caption("Mechanical computation — not financial advice.")

    with tab_catalysts:
        render_catalysts(result)


# ── Sidebar ──
with st.sidebar:
    st.markdown("### \U0001f4ca TalicoTrading")
    st.caption("Live data via yfinance • 5-min cache")
    st.markdown("")
    ticker_input = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").upper().strip()
    period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=0)
    analyze_btn = st.button("\U0001f50d Analyze", type="primary", use_container_width=True)
    st.divider()
    positions = load_positions()
    if positions:
        st.caption(f"\U0001f4bc {len(positions)} open position{'s' if len(positions) > 1 else ''}")
    wl_count = len(load_watchlist())
    if wl_count:
        st.caption(f"\U0001f4cb {wl_count} watchlist ticker{'s' if wl_count > 1 else ''}")
    fx = cached_fx_rate()
    st.caption(f"\U0001f4b1 USD/CAD: {fx}")

# ── Main ──
st.caption("⚠️ Informational and educational only. Not financial advice. Always do your own research.")

tab_analyze, tab_watchlist, tab_positions, tab_movers, tab_screener, tab_history = st.tabs(
    ["\U0001f50d Analyze", "\U0001f4cb Watchlist", "\U0001f4bc Positions",
     "\U0001f525 Movers", "\U0001f4e1 Screener", "\U0001f4ca History"])


# === ANALYZE ===
with tab_analyze:
    if analyze_btn and ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            result = cached_analyze(ticker_input, period)
        render_analysis(result)
    elif analyze_btn:
        st.warning("Please enter a ticker symbol.")
    else:
        st.markdown("")
        st.markdown("#### Enter a ticker and click Analyze")
        st.markdown("Get a clear answer: **should you buy this stock today?**")
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**📝 Verdict**\n\nBuy now / Wait / Avoid with full reasoning")
        c2.markdown("**\U0001f527 Indicators**\n\nEach one explained: what it means for buying now")
        c3.markdown("**\U0001f4f0 Sentiment**\n\nRecent headlines scored with impact analysis")
        c4.markdown("**\U0001f680 Catalysts**\n\nAnalyst targets, earnings, insider activity")


# === WATCHLIST ===
with tab_watchlist:
    wl_tickers = load_watchlist()

    ac, mc = st.columns([1, 2])
    with ac:
        with st.form("add_wl", clear_on_submit=True):
            wl_new = st.text_input("Add Ticker", placeholder="e.g. TSLA").upper().strip()
            if st.form_submit_button("Add to Watchlist", use_container_width=True, type="primary") and wl_new:
                add_to_watchlist(wl_new)
                st.rerun()
    with mc:
        if wl_tickers:
            st.markdown(f"**Tracking {len(wl_tickers)} tickers**")
            tag_cols = st.columns(min(len(wl_tickers), 10))
            for i, t in enumerate(wl_tickers):
                with tag_cols[i % min(len(wl_tickers), 10)]:
                    if st.button(f"✕ {t}", key=f"wl_rm_{t}"):
                        remove_from_watchlist(t)
                        st.rerun()

    if not wl_tickers:
        st.markdown("---")
        st.markdown("Add tickers above to build your watchlist.")
    else:
        st.divider()
        if st.button("Analyze All Watchlist", type="primary", use_container_width=True):
            progress = st.progress(0, text="Analyzing...")
            results = []
            for i, t in enumerate(wl_tickers):
                r = cached_analyze(t, period)
                results.append(r)
                progress.progress((i + 1) / len(wl_tickers), text=f"Analyzing {t}...")
            progress.empty()

            valid_results = [r for r in results if not r.get("error")]
            error_results = [r for r in results if r.get("error")]
            valid_results.sort(key=lambda r: r["rating"]["combined_score"], reverse=True)

            if valid_results:
                avg_score = sum(r["rating"]["combined_score"] for r in valid_results) / len(valid_results)
                buy_now = sum(1 for r in valid_results if r.get("buy_timing", {}).get("timing") == "Buy Now")
                bulls = sum(1 for r in valid_results if r["rating"]["rating"] in ("Buy", "Strong Buy"))
                bears = sum(1 for r in valid_results if r["rating"]["rating"] in ("Sell", "Strong Sell"))
                aligned = sum(1 for r in valid_results if r.get("sentiment_technical_agreement") == "aligned")

                oc1, oc2, oc3, oc4, oc5 = st.columns(5)
                oc1.metric("Avg Score", f"{avg_score:.0f}/100")
                oc2.metric("Buy Now", buy_now)
                oc3.metric("Bullish", bulls)
                oc4.metric("Bearish", bears)
                oc5.metric("High Conviction", aligned)

            # Overview table
            rows = []
            for r in valid_results:
                tech = r["technicals"]
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, r["rating"]["combined_score"])
                bt = r.get("buy_timing", {})
                rows.append({
                    "Ticker": r["ticker"],
                    "Price (C$)": f"{usd_to_cad(tech['current_price']):.2f}",
                    "Change": f"{'+' if tech['price_change_pct'] >= 0 else ''}{tech['price_change_pct']}%",
                    "Timing": bt.get("timing", "—"),
                    "Rating": f"{r['rating']['rating']}{arr}",
                    "Score": r["rating"]["combined_score"],
                    "RSI": tech["rsi"],
                    "Trend": tech["trend"].title(),
                    "Sentiment": r["sentiment"]["overall_sentiment"].title(),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.divider()

            # Per-ticker expanded cards — show ALL details on expand
            for r in valid_results:
                if r.get("error"):
                    continue
                tech = r["technicals"]
                sent = r["sentiment"]
                rating = r["rating"]
                bt = r.get("buy_timing", {})
                wa = r.get("written_analysis", {})
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, rating["combined_score"])
                rc = RATING_COLORS.get(rating["rating"], "#757575")
                tc = TIMING_COLORS.get(bt.get("timing", ""), "#757575")

                label = (f"{r['ticker']} — {bt.get('timing', '?')} | "
                         f"{rating['rating']}{arr} ({rating['combined_score']:.0f}/100)")

                with st.expander(label, expanded=False):
                    # Header row
                    wc1, wc2, wc3 = st.columns([3, 1.5, 1.5])
                    with wc1:
                        price_color = "#00c853" if tech["price_change_pct"] >= 0 else "#ff5252"
                        st.markdown(
                            f"**{r['ticker']} — {r['info']['shortName']}** &nbsp; "
                            f"<span style='font-size:1.2em;font-weight:700'>C${usd_to_cad(tech['current_price']):.2f}</span> "
                            f"<span style='color:{price_color}'>"
                            f"({'+' if tech['price_change_pct'] >= 0 else ''}{tech['price_change_pct']}%)</span>",
                            unsafe_allow_html=True)
                    with wc2:
                        st.markdown(f"<div class='timing-badge' style='background:{tc};color:#000;font-size:0.9em;padding:6px 10px'>"
                                    f"{bt.get('timing', '?')}</div>", unsafe_allow_html=True)
                    with wc3:
                        st.markdown(f"<div class='rating-badge' style='background:{rc};padding:6px 10px'>"
                                    f"<div class='label' style='font-size:1em'>{rating['rating']}{arr}</div>"
                                    f"<div class='score' style='font-size:0.9em'>{rating['combined_score']:.0f}/100</div></div>",
                                    unsafe_allow_html=True)

                    # Quick metrics
                    qm1, qm2, qm3, qm4, qm5 = st.columns(5)
                    qm1.metric("RSI", f"{tech['rsi']:.0f}",
                               "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
                    qm2.metric("Trend", tech['trend'].title())
                    qm3.metric("Vol", f"{tech['vol_ratio']:.1f}x")
                    qm4.metric("5d", f"{tech.get('ret_5d', 0):+.1f}%")
                    agr = r.get("sentiment_technical_agreement", "neutral")
                    qm5.metric("Agreement", agr.title())

                    # Written summary
                    st.markdown("---")
                    st.markdown(wa.get("summary", ""))

                    # Bull / Bear signals side by side
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown('<p class="section-header">\U0001f7e2 Bullish</p>', unsafe_allow_html=True)
                        for sig in wa.get("bull_signals", [])[:4]:
                            st.markdown(f"<div style='font-size:0.85em;padding:3px 0'>• {sig}</div>", unsafe_allow_html=True)
                    with bc2:
                        st.markdown('<p class="section-header">\U0001f534 Bearish</p>', unsafe_allow_html=True)
                        for sig in wa.get("bear_signals", [])[:4]:
                            st.markdown(f"<div style='font-size:0.85em;padding:3px 0'>• {sig}</div>", unsafe_allow_html=True)

                    # Risks + Better entry
                    if wa.get("risks"):
                        st.markdown('<p class="section-header">⚠️ Risks</p>', unsafe_allow_html=True)
                        for risk in wa["risks"][:3]:
                            st.markdown(f"<div style='font-size:0.85em'>• {risk}</div>", unsafe_allow_html=True)

                    if bt.get("better_entry"):
                        st.markdown('<p class="section-header">\U0001f4a1 Better Entry</p>', unsafe_allow_html=True)
                        for tip in bt["better_entry"][:2]:
                            st.markdown(f"<div style='font-size:0.85em'>• {tip}</div>", unsafe_allow_html=True)

                    # News sentiment summary
                    st.markdown("---")
                    sent_c = "#00c853" if sent["overall_sentiment"] == "bullish" else "#ff5252" if sent["overall_sentiment"] == "bearish" else "#ffd600"
                    st.markdown(f"**Sentiment:** <span style='color:{sent_c}'>{sent['overall_sentiment'].title()}</span> "
                                f"({sent.get('strength', 'none')}) — "
                                f"\U0001f7e2 {sent['bullish_count']} ⚪ {sent['neutral_count']} \U0001f534 {sent['bearish_count']}",
                                unsafe_allow_html=True)

                    # Support/Resistance
                    sr1, sr2 = st.columns(2)
                    with sr1:
                        if tech["support_levels"]:
                            st.markdown(f"**Support:** C${usd_to_cad(tech['support_levels'][0])}")
                        else:
                            st.caption("No support detected")
                    with sr2:
                        if tech["resistance_levels"]:
                            st.markdown(f"**Resistance:** C${usd_to_cad(tech['resistance_levels'][0])}")
                        else:
                            st.caption("No resistance detected")

                    # Rating history
                    render_rating_history(r)

            if error_results:
                st.divider()
                for r in error_results:
                    st.error(f"**{r['ticker']}:** {r['error']}")

            csv_rows = list(rows) if rows else []
            if csv_rows:
                st.download_button("Download CSV", pd.DataFrame(csv_rows).to_csv(index=False),
                                   "watchlist_analysis.csv", "text/csv")


# === POSITIONS ===
with tab_positions:
    st.markdown('<p class="section-header">Add New Position</p>', unsafe_allow_html=True)
    with st.form("add_pos", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        pos_ticker = fc1.text_input("Ticker", placeholder="e.g. AAPL").upper().strip()
        pos_entry = fc2.number_input("Entry Price (USD)", min_value=0.01, step=0.01, format="%.2f")
        pos_shares = fc3.number_input("Shares", min_value=0.0001, step=1.0, format="%.4f", value=1.0)
        fc4, fc5, fc6 = st.columns(3)
        pos_date = fc4.date_input("Entry Date", value=date.today())
        pos_sl = fc5.number_input("Stop Loss (USD)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
        pos_tp = fc6.number_input("Target Price (USD)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
        pos_notes = st.text_input("Notes", placeholder="e.g. Bought on SMA20 bounce")
        if st.form_submit_button("Add Position", type="primary", use_container_width=True):
            if pos_ticker and pos_entry > 0:
                sent_score = None
                try:
                    quick = cached_analyze(pos_ticker, "3mo")
                    if not quick.get("error"):
                        sent_score = quick["sentiment"]["sentiment_score"]
                except Exception:
                    pass
                add_position(pos_ticker, pos_entry, pos_shares, pos_date.strftime("%Y-%m-%d"),
                             pos_sl if pos_sl > 0 else None, pos_tp if pos_tp > 0 else None,
                             pos_notes, entry_sentiment_score=sent_score)
                st.rerun()

    st.divider()
    positions = load_positions()
    if not positions:
        st.markdown("*No positions yet. Add one above to start tracking.*")
    else:
        st.markdown(f"#### Open Positions ({len(positions)})")
        pos_results = []
        with st.spinner("Analyzing positions..."):
            for pos in positions:
                a = cached_analyze(pos["ticker"], "6mo")
                g = analyze_position(pos, a, cached_fx_rate())
                pos_results.append((pos, a, g))

        total_cost = sum(g["cost_basis"] for _, _, g in pos_results if "cost_basis" in g)
        total_val = sum(g["market_value"] for _, _, g in pos_results if "market_value" in g)
        total_pnl = total_val - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Cost Basis", f"C${usd_to_cad(total_cost):,.2f}")
        sc2.metric("Market Value", f"C${usd_to_cad(total_val):,.2f}")
        pfx = "+" if total_pnl >= 0 else ""
        sc3.metric("Total P&L", f"{pfx}C${usd_to_cad(total_pnl):,.2f}", f"{pfx}{total_pnl_pct:.1f}%")
        sc4.metric("Positions", len(positions))

        summary = []
        for pos, a, g in pos_results:
            pnl_s = f"{'+' if g['pnl_pct'] >= 0 else ''}{g['pnl_pct']}%"
            hist = a.get("rating_history", []) if not a.get("error") else []
            score_now = a["rating"]["combined_score"] if not a.get("error") else 0
            arr = trend_arrow(hist, score_now)
            bt = a.get("buy_timing", {}) if not a.get("error") else {}
            summary.append({
                "Ticker": pos["ticker"],
                "Entry (C$)": f"{usd_to_cad(pos['entry_price']):.2f}",
                "Now (C$)": f"{usd_to_cad(g['current_price']):.2f}",
                "P&L": f"{'+' if g['pnl_total'] >= 0 else ''}{usd_to_cad(g['pnl_total']):.2f} ({pnl_s})",
                "Days": g["days_held"],
                "Rating": f"{a['rating']['rating']}{arr}" if not a.get("error") else "N/A",
                "Action": g["action"],
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.divider()

        for pos, a, g in pos_results:
            ac_color = ACTION_COLORS.get(g["action"], "#757575")
            label = f"{pos['ticker']} — {g['action']} ({'+' if g['pnl_pct'] >= 0 else ''}{g['pnl_pct']}%)"
            with st.expander(label, expanded=(g["urgency"] == "high")):
                phc1, phc2, phc3 = st.columns([2, 1, 1])
                with phc1:
                    st.markdown(f"**Entry:** C${usd_to_cad(pos['entry_price']):.2f} on {pos['entry_date']} ({pos['shares']} shares)")
                    if pos.get("stop_loss"):
                        st.markdown(f"**Stop:** C${usd_to_cad(pos['stop_loss']):.2f}")
                    elif g.get("atr_suggested_stop"):
                        st.markdown(f"**Suggested Stop (2×ATR):** C${usd_to_cad(g['atr_suggested_stop']):.2f}")
                    if pos.get("target_price"):
                        st.markdown(f"**Target:** C${usd_to_cad(pos['target_price']):.2f}")
                    elif g.get("atr_suggested_target"):
                        st.markdown(f"**Suggested Target (3×ATR):** C${usd_to_cad(g['atr_suggested_target']):.2f}")
                    if g.get("atr_trailing_stop"):
                        st.markdown(f"**Trail (1.5×ATR):** C${usd_to_cad(g['atr_trailing_stop']):.2f}")
                    if pos.get("notes"):
                        st.caption(pos["notes"])
                with phc2:
                    pfx = "+" if g["pnl_pct"] >= 0 else ""
                    st.metric("P&L", f"{pfx}C${usd_to_cad(g['pnl_total']):.2f}", f"{pfx}{g['pnl_pct']:.1f}%")
                with phc3:
                    st.markdown(
                        f"<div class='rating-badge' style='background:{ac_color}'>"
                        f"<div class='label'>{g['action']}</div></div>",
                        unsafe_allow_html=True)

                dm1, dm2, dm3, dm4, dm5 = st.columns(5)
                if not a.get("error"):
                    tech = a["technicals"]
                    dm1.metric("RSI", f"{tech['rsi']:.0f}")
                    dm2.metric("Trend", tech["trend"].title())
                    dm3.metric("Rating", a["rating"]["rating"], f"{a['rating']['combined_score']:.0f}")
                sw = 15
                dr = max(0, sw - g["trading_days_held"])
                dm4.metric("Days", f"{g['days_held']}d", f"{dr} trading days left" if dr > 0 else "Past window")
                if pos.get("stop_loss") and pos.get("target_price"):
                    risk = abs(pos["entry_price"] - pos["stop_loss"])
                    reward = abs(pos["target_price"] - pos["entry_price"])
                    rr = round(reward / risk, 2) if risk > 0 else 0
                    dm5.metric("R/R", f"{rr}:1")
                else:
                    dm5.metric("R/R", "N/A")

                if not a.get("error"):
                    sent = a.get("sentiment", {})
                    if sent.get("has_news"):
                        sc = "#00c853" if sent["overall_sentiment"] == "bullish" else "#ff5252" if sent["overall_sentiment"] == "bearish" else "#ffd600"
                        st.markdown(f"**Sentiment:** <span style='color:{sc}'>{sent['overall_sentiment'].title()}</span> "
                                    f"({sent.get('strength', '')}) — {sent['significance'][:120]}",
                                    unsafe_allow_html=True)
                    entry_sent = pos.get("entry_sentiment_score")
                    if entry_sent is not None and sent.get("has_news"):
                        now_sent = sent["sentiment_score"]
                        shift = now_sent - entry_sent
                        if abs(shift) > 5:
                            shift_c = "#00c853" if shift > 0 else "#ff5252"
                            st.markdown(f"**Sentiment Shift:** <span style='color:{shift_c}'>"
                                        f"{entry_sent:.0f} → {now_sent:.0f} ({shift:+.0f})</span>",
                                        unsafe_allow_html=True)

                    earnings = a.get("earnings", {})
                    if earnings.get("has_date") and earnings.get("in_swing_window"):
                        eps_str = f" | EPS Est: ${earnings['eps_estimate']:.2f}" if earnings.get("eps_estimate") is not None else ""
                        st.warning(f"⚠️ Earnings in {earnings['days_until']} days ({earnings['date_str']}){eps_str}")

                st.markdown("---")
                st.markdown('<p class="section-header">Guidance</p>', unsafe_allow_html=True)
                for reason in g["reasons"]:
                    icon = "→"
                    if "STOP LOSS" in reason or "TARGET REACHED" in reason or "EARNINGS" in reason:
                        icon = "\U0001f6a8"
                    elif "bullish" in reason.lower() or "support" in reason.lower():
                        icon = "\U0001f7e2"
                    elif "bearish" in reason.lower() or "overbought" in reason.lower() or "loss" in reason.lower():
                        icon = "\U0001f534"
                    st.write(f"{icon} {reason}")

                if not a.get("error") and "df" in a:
                    adf = a["df"].copy()
                    fv = cached_fx_rate()
                    adf["Close_CAD"] = adf["Close"] * fv
                    entry_cad = pos["entry_price"] * fv
                    mini = go.Figure()
                    mini.add_trace(go.Scatter(x=adf.index, y=adf["Close_CAD"], mode="lines",
                                              line=dict(color="#42a5f5", width=2)))
                    mini.add_hline(y=entry_cad, line_dash="dot", line_color="#ffd600", line_width=2,
                                   annotation_text=f"Entry C${entry_cad:.2f}")
                    sl = pos.get("stop_loss") or g.get("atr_suggested_stop")
                    tp = pos.get("target_price") or g.get("atr_suggested_target")
                    if sl:
                        mini.add_hline(y=sl * fv, line_dash="dash", line_color="#ff5252", line_width=1,
                                       annotation_text=f"Stop C${sl * fv:.2f}")
                    if tp:
                        mini.add_hline(y=tp * fv, line_dash="dash", line_color="#00c853", line_width=1,
                                       annotation_text=f"Target C${tp * fv:.2f}")
                    mini.update_layout(template="plotly_dark", height=220,
                                       margin=dict(l=50, r=50, t=10, b=30),
                                       yaxis_title="C$", showlegend=False)
                    st.plotly_chart(mini, use_container_width=True, key=_next_key("pos_mini"))

                st.markdown("---")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    with st.form(f"edit_{pos['id']}"):
                        st.markdown("**Edit**")
                        new_sl = st.number_input("Stop (USD)", value=pos.get("stop_loss") or 0.0,
                                                  min_value=0.0, step=0.01, format="%.2f", key=f"sl_{pos['id']}")
                        new_tp = st.number_input("Target (USD)", value=pos.get("target_price") or 0.0,
                                                  min_value=0.0, step=0.01, format="%.2f", key=f"tp_{pos['id']}")
                        new_notes = st.text_input("Notes", value=pos.get("notes", ""), key=f"nt_{pos['id']}")
                        if st.form_submit_button("Update"):
                            update_position(pos["id"],
                                            stop_loss=new_sl if new_sl > 0 else None,
                                            target_price=new_tp if new_tp > 0 else None,
                                            notes=new_notes)
                            st.rerun()
                with ec2:
                    with st.form(f"close_{pos['id']}"):
                        st.markdown("**Close Position**")
                        exit_price = st.number_input("Exit Price (USD)", min_value=0.01, step=0.01,
                                                      value=float(g["current_price"]), format="%.2f",
                                                      key=f"ex_{pos['id']}")
                        if st.form_submit_button("Close & Save"):
                            close_position_with_history(pos["id"], exit_price)
                            st.rerun()
                with ec3:
                    st.markdown("**Remove**")
                    st.caption("Without saving to history")
                    if st.button("Remove", key=f"del_{pos['id']}"):
                        remove_position(pos["id"])
                        st.rerun()


# === MARKET MOVERS ===
with tab_movers:
    st.markdown("#### \U0001f525 Market Movers")
    st.caption("Scans 45+ stocks for high-impact headlines (sentiment ≥ 0.4)")

    if st.button("\U0001f525 Scan for Market Movers", type="primary", use_container_width=True):
        with st.spinner("Scanning headlines..."):
            movers = cached_market_movers()

        if not movers:
            st.info("No high-impact stories found right now.")
        else:
            bull_movers = [m for m in movers if m["sentiment"] == "bullish"]
            bear_movers = [m for m in movers if m["sentiment"] == "bearish"]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Big Stories", len(movers))
            mc2.metric("Bullish", len(bull_movers))
            mc3.metric("Bearish", len(bear_movers))

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown('<p class="section-header">\U0001f7e2 Bullish</p>', unsafe_allow_html=True)
                for m in bull_movers[:10]:
                    st.markdown(
                        f"<div class='news-card' style='border-left:3px solid #00c853'>"
                        f"<b>{m['ticker']}</b> — {m['title'][:80]}<br>"
                        f"<span style='opacity:0.5;font-size:0.85em'>{m['publisher']}</span> &nbsp;"
                        f"<span style='color:#00c853;font-weight:600'>+{int(m['compound']*100)}%</span></div>",
                        unsafe_allow_html=True)
                if not bull_movers:
                    st.caption("No bullish movers")

            with bc2:
                st.markdown('<p class="section-header">\U0001f534 Bearish</p>', unsafe_allow_html=True)
                for m in bear_movers[:10]:
                    st.markdown(
                        f"<div class='news-card' style='border-left:3px solid #ff5252'>"
                        f"<b>{m['ticker']}</b> — {m['title'][:80]}<br>"
                        f"<span style='opacity:0.5;font-size:0.85em'>{m['publisher']}</span> &nbsp;"
                        f"<span style='color:#ff5252;font-weight:600'>-{int(abs(m['compound'])*100)}%</span></div>",
                        unsafe_allow_html=True)
                if not bear_movers:
                    st.caption("No bearish movers")

            st.divider()
            mover_rows = [{"Ticker": m["ticker"], "Headline": m["title"][:80],
                           "Sentiment": m["sentiment"].title(), "Score": m["compound"],
                           "Source": m["publisher"]} for m in movers]
            st.dataframe(pd.DataFrame(mover_rows), hide_index=True, use_container_width=True)
            st.download_button("Download CSV", pd.DataFrame(mover_rows).to_csv(index=False),
                               "market_movers.csv", "text/csv")
    else:
        st.markdown("Hit the button to scan 45+ stocks for the biggest sentiment-moving headlines.")


# === SCREENER ===
with tab_screener:
    st.markdown("#### \U0001f4e1 Stock Screener")

    fc1, fc2, fc3, fc4 = st.columns(4)
    min_score = fc1.slider("Min Score", 0, 100, 0)
    setup_filter = fc2.selectbox("Setup Type", [
        "All",
        "Buy Now",
        "Pullback Entry",
        "Oversold Recovery",
        "Strong Momentum",
        "Near Support",
        "High Relative Volume",
        "Positive Sentiment",
        "High Conviction",
        "Overextended / Avoid",
    ])
    trend_filter = fc3.selectbox("Trend", ["All", "Bullish", "Bearish"])
    rsi_filter = fc4.selectbox("RSI Zone", ["All", "Oversold (<35)", "Overbought (>65)"])

    if st.button("\U0001f4e1 Run Screener", type="primary", use_container_width=True):
        progress = st.progress(0, text="Scanning...")
        screener_results = []
        for i, t in enumerate(SCREENER_TICKERS):
            try:
                r = cached_analyze(t, "3mo")
                if not r.get("error"):
                    screener_results.append(r)
            except Exception:
                pass
            progress.progress((i + 1) / len(SCREENER_TICKERS), text=f"Scanning {t}...")
        progress.empty()

        filtered = []
        for r in screener_results:
            if r["rating"]["combined_score"] < min_score:
                continue
            tech = r["technicals"]
            bt = r.get("buy_timing", {})
            timing = bt.get("timing", "")

            if rsi_filter == "Oversold (<35)" and tech["rsi"] >= 35:
                continue
            if rsi_filter == "Overbought (>65)" and tech["rsi"] <= 65:
                continue
            if trend_filter == "Bullish" and "bullish" not in tech["trend"]:
                continue
            if trend_filter == "Bearish" and "bearish" not in tech["trend"]:
                continue

            if setup_filter == "Buy Now" and timing != "Buy Now":
                continue
            elif setup_filter == "Pullback Entry" and timing != "Buy — Pullback Entry":
                continue
            elif setup_filter == "Oversold Recovery" and tech["rsi"] >= 35:
                continue
            elif setup_filter == "Strong Momentum":
                if tech.get("ret_5d", 0) < 2 or tech["rsi"] < 50:
                    continue
            elif setup_filter == "Near Support":
                if not tech["support_levels"]:
                    continue
                sup_dist = ((tech["current_price"] - tech["support_levels"][0]) / tech["current_price"]) * 100
                if sup_dist > 3:
                    continue
            elif setup_filter == "High Relative Volume" and tech["vol_ratio"] < 1.3:
                continue
            elif setup_filter == "Positive Sentiment":
                if r["sentiment"]["overall_sentiment"] != "bullish":
                    continue
            elif setup_filter == "High Conviction":
                if r.get("sentiment_technical_agreement") != "aligned":
                    continue
            elif setup_filter == "Overextended / Avoid":
                if timing not in ("Overextended", "Avoid for Now"):
                    continue

            filtered.append(r)

        filtered.sort(key=lambda r: r["rating"]["combined_score"], reverse=True)

        if filtered:
            st.subheader(f"{len(filtered)} stocks match")
            scr_rows = []
            for r in filtered:
                tech = r["technicals"]
                bt = r.get("buy_timing", {})
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, r["rating"]["combined_score"])
                scr_rows.append({
                    "Ticker": r["ticker"],
                    "Name": r["info"]["shortName"],
                    "Price (C$)": f"{usd_to_cad(tech['current_price']):.2f}",
                    "Timing": bt.get("timing", "—"),
                    "Rating": f"{r['rating']['rating']}{arr}",
                    "Score": r["rating"]["combined_score"],
                    "RSI": f"{tech['rsi']:.0f}",
                    "5d": f"{tech.get('ret_5d', 0):+.1f}%",
                    "Trend": tech["trend"].title(),
                    "Vol": f"{tech['vol_ratio']:.1f}x",
                    "Sentiment": r["sentiment"]["overall_sentiment"].title(),
                })
            st.dataframe(pd.DataFrame(scr_rows), hide_index=True, use_container_width=True)

            st.download_button("Download CSV", pd.DataFrame(scr_rows).to_csv(index=False),
                               "screener_results.csv", "text/csv")

            wl = load_watchlist()
            add_cols = st.columns(min(len(filtered), 8))
            for i, r in enumerate(filtered[:8]):
                with add_cols[i]:
                    if r["ticker"] not in wl:
                        if st.button(f"+ {r['ticker']}", key=f"scr_add_{r['ticker']}"):
                            add_to_watchlist(r["ticker"])
                            st.rerun()

            st.divider()
            for r in filtered[:10]:
                bt = r.get("buy_timing", {})
                tc = TIMING_COLORS.get(bt.get("timing", ""), "#757575")
                with st.expander(f"{r['ticker']} — {bt.get('timing', '?')} | {r['rating']['rating']} ({r['rating']['combined_score']:.0f})"):
                    render_analysis(r)
        else:
            st.warning("No stocks match your filters. Try widening your criteria.")
    else:
        st.markdown("Use the filters above and click **Run Screener** to find stocks worth buying now.")
        st.markdown("")
        st.markdown("**Setup types explained:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**Buy Now** — Multiple buy signals confirmed\n\n"
                    "**Pullback Entry** — Uptrend with recent dip")
        c2.markdown("**Oversold Recovery** — RSI below 35\n\n"
                    "**Strong Momentum** — 5-day gain with rising RSI")
        c3.markdown("**Near Support** — Within 3% of support\n\n"
                    "**High Rel Volume** — Volume 1.3x+ average")
        c4.markdown("**Positive Sentiment** — Bullish news flow\n\n"
                    "**Overextended** — Avoid — too stretched")


# === TRADE HISTORY ===
with tab_history:
    st.markdown("#### \U0001f4ca Trade History")
    stats = compute_trade_statistics()
    if not stats["has_data"]:
        st.markdown("No closed trades yet. Close a position from **Positions** to start tracking performance.")
    else:
        history = stats["history"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Trades", stats["total_trades"])
        mc2.metric("Win Rate", f"{stats['win_rate']}%", f"{stats['wins']}W / {stats['losses']}L")
        mc3.metric("Profit Factor", f"{stats['profit_factor']}")
        avg_all = sum(t["pnl_pct"] for t in history) / len(history) if history else 0
        mc4.metric("Avg Trade", f"{avg_all:+.2f}%")
        mc5.metric("Total P&L", f"C${usd_to_cad(stats['total_pnl']):,.2f}")

        mc6, mc7, mc8 = st.columns(3)
        mc6.metric("Avg Win", f"{stats['avg_gain']:+.2f}%")
        mc7.metric("Avg Loss", f"{stats['avg_loss']:+.2f}%")
        mc8.metric("Avg Hold", f"{stats['avg_hold_days']} days")

        st.divider()
        wc1, wc2 = st.columns(2)
        with wc1:
            wins = [t for t in history if t["pnl_pct"] >= 0]
            st.markdown(f'<p class="section-header">\U0001f7e2 Wins ({len(wins)})</p>', unsafe_allow_html=True)
            if wins:
                win_rows = [{"Ticker": t["ticker"], "Entry": f"C${usd_to_cad(t['entry_price']):.2f}",
                             "Exit": f"C${usd_to_cad(t['exit_price']):.2f}",
                             "P&L": f"+{t['pnl_pct']:.2f}%", "Held": f"{t['days_held']}d"}
                            for t in sorted(wins, key=lambda x: x["pnl_pct"], reverse=True)]
                st.dataframe(pd.DataFrame(win_rows), hide_index=True, use_container_width=True)
        with wc2:
            losses = [t for t in history if t["pnl_pct"] < 0]
            st.markdown(f'<p class="section-header">\U0001f534 Losses ({len(losses)})</p>', unsafe_allow_html=True)
            if losses:
                loss_rows = [{"Ticker": t["ticker"], "Entry": f"C${usd_to_cad(t['entry_price']):.2f}",
                              "Exit": f"C${usd_to_cad(t['exit_price']):.2f}",
                              "P&L": f"{t['pnl_pct']:.2f}%", "Held": f"{t['days_held']}d"}
                             for t in sorted(losses, key=lambda x: x["pnl_pct"])]
                st.dataframe(pd.DataFrame(loss_rows), hide_index=True, use_container_width=True)

        st.divider()
        trade_labels = [f"{t['ticker']}\n{t['exit_date']}" for t in history]
        trade_pcts = [t["pnl_pct"] for t in history]
        bar_colors = ["#00c853" if p >= 0 else "#ff5252" for p in trade_pcts]

        fig_trades = go.Figure()
        fig_trades.add_trace(go.Bar(x=trade_labels, y=trade_pcts, marker_color=bar_colors,
                                     text=[f"{p:+.1f}%" for p in trade_pcts], textposition="outside"))
        fig_trades.add_hline(y=avg_all, line_dash="dash", line_color="#42a5f5",
                             annotation_text=f"Avg: {avg_all:+.1f}%")
        fig_trades.update_layout(template="plotly_dark", height=320, yaxis_title="P&L %",
                                  margin=dict(l=50, r=50, t=20, b=80), showlegend=False)
        st.plotly_chart(fig_trades, use_container_width=True, key=_next_key("trades_bar"))

        cum_pnl = []
        running = 0
        for t in history:
            running += t["pnl_total"]
            cum_pnl.append(running)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=[t["exit_date"] for t in history], y=[usd_to_cad(p) for p in cum_pnl],
                                     mode="lines+markers", fill="tozeroy",
                                     line=dict(color="#42a5f5", width=2), fillcolor="rgba(66,165,245,0.1)"))
        fig_eq.update_layout(template="plotly_dark", height=250, yaxis_title="Cumulative P&L (C$)",
                              margin=dict(l=50, r=50, t=20, b=30))
        st.plotly_chart(fig_eq, use_container_width=True, key=_next_key("equity_curve"))

        hist_rows = [{"Ticker": t["ticker"],
                      "Entry": f"C${usd_to_cad(t['entry_price']):.2f}",
                      "Exit": f"C${usd_to_cad(t['exit_price']):.2f}",
                      "P&L %": f"{t['pnl_pct']:+.2f}%",
                      "P&L": f"C${usd_to_cad(t['pnl_total']):,.2f}",
                      "Held": f"{t['days_held']}d",
                      "Entry Date": t["entry_date"],
                      "Exit Date": t["exit_date"]}
                     for t in history]
        st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
        st.download_button("Download Trade History", pd.DataFrame(hist_rows).to_csv(index=False),
                           "trade_history.csv", "text/csv")
