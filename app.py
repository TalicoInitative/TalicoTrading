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

st.set_page_config(page_title="Swing Trading Assistant", page_icon="\U0001f4ca",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Global font and spacing */
[data-testid="stAppViewContainer"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
section[data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid #1e2530; }
section[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; letter-spacing: -0.02em; }

/* Tabs styling */
button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 500 !important; padding: 10px 16px !important; }
button[data-baseweb="tab"][aria-selected="true"] { font-weight: 700 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: 700 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }

/* Expanders */
[data-testid="stExpander"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    margin-bottom: 8px;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; }

/* Primary buttons — vibrant blue, always visible */
button[data-testid="stBaseButton-primary"] {
    border-radius: 8px !important; font-weight: 700 !important;
    letter-spacing: 0.02em;
    background-color: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: background-color 0.2s !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1565c0 !important;
}
button[data-testid="stBaseButton-primary"]:active {
    background-color: #0d47a1 !important;
}

/* Dividers */
hr { border-color: #21262d !important; opacity: 0.5 !important; }

/* Alert / info boxes */
[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid #21262d !important; }

/* Download buttons */
button[data-testid="stBaseButton-secondary"] { border-radius: 8px !important; }

/* Form containers */
[data-testid="stForm"] { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }

/* Rating badge helper */
.rating-badge {
    display: inline-block; padding: 14px 20px; border-radius: 12px;
    text-align: center; font-weight: 700; line-height: 1.3;
}
.rating-badge .label { font-size: 1.3em; color: #000; }
.rating-badge .score { font-size: 1.1em; color: #000; }

/* Sentiment inline badge */
.sent-badge {
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-weight: 600; font-size: 0.85em;
}

/* Section headers */
.section-header {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #8b949e; font-weight: 600; margin-bottom: 8px; margin-top: 16px;
}

/* Card container */
.card {
    background: #161b22; border: 1px solid #21262d; border-radius: 12px;
    padding: 16px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

RATING_COLORS = {
    "Strong Buy": "#00c853", "Buy": "#69f0ae", "Neutral": "#ffd600",
    "Sell": "#ff5252", "Strong Sell": "#d50000",
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

def render_rating_history(result):
    history = result.get("rating_history", [])
    if not history:
        st.caption("Not enough data for rating history.")
        return

    chart_key = _next_key("rh")

    current = result["rating"]
    all_days = list(history) + [{
        "date": "Today",
        "score": current["combined_score"],
        "rating": current["rating"],
        "component_scores": current["component_scores"],
        "change_explanation": None,
    }]

    if len(history) > 0:
        today_prev = history[-1]
        all_days[-1]["change_explanation"] = explain_rating_change(today_prev, {
            "score": current["combined_score"], "rating": current["rating"],
            "component_scores": current["component_scores"],
        })

    # Reverse order: Today on LEFT, oldest on RIGHT
    all_days_reversed = list(reversed(all_days))

    st.markdown('<p class="section-header">Rating History — Last 5 Days + Today</p>', unsafe_allow_html=True)
    st.caption("Most recent on the left, oldest on the right")

    cols = st.columns(len(all_days_reversed))
    for i, day in enumerate(all_days_reversed):
        rc = RATING_COLORS.get(day["rating"], "#757575")
        is_today = day["date"] == "Today"
        border = "3px solid rgba(255,255,255,0.8)" if is_today else "1px solid #333"
        shadow = "0 0 12px rgba(255,255,255,0.15)" if is_today else "none"
        date_label = day["date"] if not is_today else "Today"
        with cols[i]:
            st.markdown(
                f"<div style='background:{rc};color:#000;padding:10px 6px;border-radius:10px;"
                f"text-align:center;font-size:0.82em;border:{border};box-shadow:{shadow};'>"
                f"<div style='font-weight:700;font-size:0.9em;margin-bottom:2px'>{date_label}</div>"
                f"<div style='opacity:0.8;font-size:0.85em'>{day['rating']}</div>"
                f"<div style='font-size:1.4em;font-weight:800;margin-top:2px'>"
                f"{day['score']:.0f}</div>"
                f"<div style='font-size:0.7em;opacity:0.6'>/100</div></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Day-by-Day Breakdown</p>', unsafe_allow_html=True)
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
        if details:
            for d in details:
                color = "#00c853" if d["delta"] > 0 else "#ff5252"
                icon = "▲" if d["delta"] > 0 else "▼"
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{color}'>{icon}</span> {d['text']}",
                    unsafe_allow_html=True)

    if len(all_days) >= 2:
        scores = [d["score"] for d in all_days]
        dates = [d["date"] for d in all_days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers+text",
                                  text=[f"{s:.0f}" for s in scores], textposition="top center",
                                  line=dict(color="#42a5f5", width=3), marker=dict(size=10)))
        fig.add_hline(y=80, line_dash="dash", line_color="#00c853", opacity=0.4, annotation_text="Strong Buy")
        fig.add_hline(y=65, line_dash="dash", line_color="#69f0ae", opacity=0.4, annotation_text="Buy")
        fig.add_hline(y=45, line_dash="dash", line_color="#ffd600", opacity=0.3)
        fig.add_hline(y=30, line_dash="dash", line_color="#ff5252", opacity=0.3)
        fig.update_layout(template="plotly_dark", height=220,
                          yaxis=dict(range=[0, 105], title="Score"),
                          margin=dict(l=50, r=50, t=10, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_position_sentiment(analysis):
    sent = analysis.get("sentiment", {})
    if not sent.get("has_news"):
        st.caption("No recent news for sentiment analysis.")
        return

    sc1, sc2, sc3 = st.columns(3)
    overall = sent["overall_sentiment"]
    color = "#00c853" if overall == "bullish" else "#ff5252" if overall == "bearish" else "#ffd600"
    with sc1:
        st.markdown(f"**Sentiment:** <span style='color:{color}'><b>{overall.title()}</b></span> "
                    f"({sent['strength']})", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"**Score:** {sent['sentiment_score']:.0f}/100 &nbsp; | &nbsp; "
                    f"\U0001f7e2 {sent['bullish_count']} &nbsp; ⚪ {sent['neutral_count']} "
                    f"&nbsp; \U0001f534 {sent['bearish_count']}")
    with sc3:
        agreement = analysis.get("sentiment_technical_agreement", "neutral")
        agr_icon = "\U0001f7e2" if agreement == "aligned" else "\U0001f534" if agreement == "divergent" else "⚪"
        agr_label = "Aligned" if agreement == "aligned" else "Divergent" if agreement == "divergent" else "Neutral"
        st.markdown(f"**Tech Agreement:** {agr_icon} {agr_label}")

    st.caption(f"\U0001f4f0 {sent['significance']}")


def render_catalysts(result):
    info = result["info"]
    analyst = result.get("analyst", {})
    insider = result.get("insider", {})
    earnings = result.get("earnings", {})
    rs = result.get("relative_strength", {})
    divergences = result.get("divergences", [])
    sentiment = result.get("sentiment", {})
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
            st.caption(f"Analysts: {info.get('numberOfAnalystOpinions', 'N/A')} | Consensus: {info.get('recommendationKey', 'N/A')}")
        else:
            st.caption("No analyst target data available")

        if analyst.get("has_data"):
            st.markdown(f"Votes: **{analyst['strong_buy']+analyst['buy']}** Buy / "
                        f"**{analyst['hold']}** Hold / **{analyst['sell']+analyst['strong_sell']}** Sell")

    with c2:
        st.markdown('<p class="section-header">Earnings</p>', unsafe_allow_html=True)
        if earnings.get("has_date"):
            days = earnings["days_until"]
            icon = "\U0001f6a8" if days <= 5 else "⚠️" if days <= 14 else "\U0001f4c5"
            st.markdown(f"{icon} Next earnings: **{earnings['date_str']}** ({days} days)")
            if earnings.get("eps_estimate") is not None:
                st.markdown(f"EPS Estimate: **${earnings['eps_estimate']:.2f}**")
            if earnings["in_swing_window"]:
                st.warning("Earnings within swing window — increased volatility risk")
        else:
            st.caption("No upcoming earnings date available")

        if earnings.get("history"):
            st.markdown('<p class="section-header">Earnings Track Record</p>', unsafe_allow_html=True)
            beat_pct = (earnings["beat_count"] / len(earnings["history"]) * 100) if earnings["history"] else 0
            avg_s = earnings.get("avg_surprise_pct", 0)
            avg_color = "#00c853" if avg_s > 0 else "#ff5252" if avg_s < 0 else "#ffd600"
            st.markdown(
                f"Last {len(earnings['history'])} quarters: "
                f"**{earnings['beat_count']}** beat, **{earnings['miss_count']}** miss, "
                f"**{earnings['meet_count']}** met "
                f"| Avg surprise: <span style='color:{avg_color}'><b>{avg_s:+.1f}%</b></span>",
                unsafe_allow_html=True)
            for eh in earnings["history"]:
                v_icon = "\U0001f7e2" if eh["verdict"] == "beat" else "\U0001f534" if eh["verdict"] == "miss" else "⚪"
                v_color = "#00c853" if eh["verdict"] == "beat" else "#ff5252" if eh["verdict"] == "miss" else "#ffd600"
                st.markdown(
                    f"&nbsp;&nbsp;{v_icon} {eh['date']}: "
                    f"Est **${eh['eps_estimate']:.2f}** → Actual **${eh['eps_actual']:.2f}** "
                    f"<span style='color:{v_color}'>({eh['surprise_pct']:+.1f}% {eh['verdict']})</span>",
                    unsafe_allow_html=True)

        st.markdown('<p class="section-header">Short Interest</p>', unsafe_allow_html=True)
        if info.get("shortPercentOfFloat"):
            spf = info["shortPercentOfFloat"] * 100
            color = "#ff5252" if spf > 10 else "#ffd600" if spf > 5 else "#00c853"
            st.markdown(f"Short % of Float: <span style='color:{color}'><b>{spf:.1f}%</b></span>",
                        unsafe_allow_html=True)
            if info.get("shortRatio"):
                st.caption(f"Short Ratio (days to cover): {info['shortRatio']:.1f}")
        else:
            st.caption("No short interest data available")

    with c3:
        st.markdown('<p class="section-header">Relative Strength vs SPY</p>', unsafe_allow_html=True)
        if rs.get("has_data"):
            color = "#00c853" if rs["rs_ratio"] > 2 else "#ff5252" if rs["rs_ratio"] < -2 else "#ffd600"
            st.markdown(f"20-day RS: <span style='color:{color}'><b>{rs['rs_ratio']:+.1f}%</b></span> "
                        f"({rs['label']})", unsafe_allow_html=True)
            st.caption(f"Stock: {rs['stock_return']:+.1f}% vs SPY: {rs['spy_return']:+.1f}%")
        else:
            st.caption("Relative strength data unavailable")

        if info.get("beta"):
            st.markdown(f"**Beta:** {info['beta']:.2f}")

    if info.get("fiftyTwoWeekHigh") and info.get("fiftyTwoWeekLow"):
        price = result["technicals"]["current_price"]
        high = info["fiftyTwoWeekHigh"]
        low = info["fiftyTwoWeekLow"]
        pos = (price - low) / (high - low) if high != low else 0.5
        st.markdown('<p class="section-header">52-Week Range</p>', unsafe_allow_html=True)
        fig_range = go.Figure(go.Bar(x=[pos * 100], y=[""], orientation="h",
                                      marker_color="#42a5f5", width=0.3))
        fig_range.update_layout(template="plotly_dark", height=60,
                                xaxis=dict(range=[0, 100], tickvals=[0, 50, 100],
                                           ticktext=[f"C${usd_to_cad(low)}", "Mid", f"C${usd_to_cad(high)}"]),
                                margin=dict(l=10, r=10, t=5, b=30), showlegend=False)
        st.plotly_chart(fig_range, use_container_width=True, key=_next_key("range"))

    st.divider()
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<p class="section-header">Insider Activity (90 days)</p>', unsafe_allow_html=True)
        if insider.get("has_data"):
            color = "#00c853" if insider["sentiment"] == "bullish" else "#ff5252" if insider["sentiment"] == "bearish" else "#ffd600"
            st.markdown(f"Sentiment: <span style='color:{color}'><b>{insider['sentiment'].title()}</b></span>",
                        unsafe_allow_html=True)
            st.caption(f"{insider['buy_count']} buys / {insider['sell_count']} sells | "
                       f"Net value: ${insider['net_value']:,.0f}")
        else:
            st.caption("No insider transaction data available")

    with ic2:
        st.markdown('<p class="section-header">Sentiment + Technical Agreement</p>', unsafe_allow_html=True)
        if agreement == "aligned":
            st.markdown("\U0001f7e2 **Aligned** — Sentiment and technicals agree. Higher conviction setup.")
        elif agreement == "divergent":
            st.markdown("\U0001f534 **Divergent** — Sentiment and technicals disagree. Potential contrarian setup or caution signal.")
        else:
            st.markdown("⚪ **Neutral** — One or both signals are neutral. No strong agreement or conflict.")

        if sentiment.get("has_news"):
            st.caption(f"Sentiment: {sentiment['overall_sentiment'].title()} ({sentiment['strength']}) | "
                       f"Technicals: {result['technicals']['trend'].title()}")

    if analyst.get("upgrades_downgrades"):
        st.markdown('<p class="section-header">Recent Analyst Actions</p>', unsafe_allow_html=True)
        ud_data = []
        for ud in analyst["upgrades_downgrades"][:5]:
            ud_data.append({"Date": ud["date"], "Firm": ud["firm"],
                            "Action": ud["action"],
                            "From": ud["from_grade"], "To": ud["to_grade"]})
        st.dataframe(pd.DataFrame(ud_data), hide_index=True, use_container_width=True)

    if divergences:
        st.markdown('<p class="section-header">Divergence Alerts</p>', unsafe_allow_html=True)
        for d in divergences:
            icon = "\U0001f7e2" if d["type"] == "bullish" else "\U0001f534"
            st.markdown(f"{icon} **{d['indicator']} {d['type'].title()} Divergence:** {d['description']}")


def render_analysis(result):
    if result.get("error"):
        st.error(result["error"])
        return

    info = result["info"]
    tech = result["technicals"]
    sent = result["sentiment"]
    rating = result["rating"]
    df = result["df"]
    fx = cached_fx_rate()
    history = result.get("rating_history", [])

    df_cad = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df_cad[col] = df_cad[col] * fx

    price_color = "green" if tech["price_change"] >= 0 else "red"
    rating_color = RATING_COLORS.get(rating["rating"], "#ffffff")
    price_cad = usd_to_cad(tech["current_price"])
    change_cad = usd_to_cad(tech["price_change"])
    arrow = trend_arrow(history, rating["combined_score"])

    st.markdown(f"### {result['ticker']} — {info['shortName']}")

    hc1, hc2, hc3, hc4 = st.columns([2.5, 1, 1, 1])
    with hc1:
        pct_prefix = "+" if tech['price_change_pct'] >= 0 else ""
        st.markdown(
            f"<div style='line-height:1.6'>"
            f"<span style='font-size:1.6em;font-weight:700'>C${price_cad}</span><br>"
            f"<span style='color:{price_color};font-size:1em;font-weight:600'>"
            f"{pct_prefix}{tech['price_change_pct']}%</span>"
            f" &nbsp;<span style='opacity:0.6'>|</span>&nbsp; "
            f"<span style='opacity:0.7'>{info['sector']}</span>"
            f"</div>", unsafe_allow_html=True)
    with hc2:
        trend_color = "#00c853" if "bullish" in tech['trend'] else "#ff5252" if "bearish" in tech['trend'] else "#ffd600"
        st.metric("Trend", tech["trend"].title())
    with hc3:
        st.metric("RSI", f"{tech['rsi']:.1f}",
                  "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
    with hc4:
        st.markdown(
            f"<div class='rating-badge' style='background:{rating_color}'>"
            f"<div class='label'>{rating['rating']}{arrow}</div>"
            f"<div class='score'>{rating['combined_score']:.0f}/100</div></div>",
            unsafe_allow_html=True)

    tab_chart, tab_tech, tab_news, tab_rating, tab_catalysts = st.tabs(
        ["\U0001f4c8 Chart", "\U0001f527 Technicals", "\U0001f4f0 News & Sentiment", "⭐ Rating", "\U0001f680 Catalysts"])

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
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                          margin=dict(l=50, r=50, t=30, b=30),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Price (C$)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True, key=_next_key("candle"))

    with tab_tech:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("RSI (14)", f"{tech['rsi']:.1f}",
                  "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
        m2.metric("MACD Hist", f"{tech['macd_hist']:.4f}",
                  "Expanding" if abs(tech["macd_hist"]) > abs(tech["macd_hist_prev"]) else "Contracting")
        m3.metric("Trend", tech["trend"].title())
        m4.metric("Vol Ratio", f"{tech['vol_ratio']}x",
                  "Above avg" if tech["vol_ratio"] > 1.2 else "Below avg" if tech["vol_ratio"] < 0.8 else "Average")
        m5.metric("ATR", f"C${usd_to_cad(tech['atr'])}", f"{tech['atr_pct']:.2f}%")

        st.markdown('<p class="section-header">Moving Averages</p>', unsafe_allow_html=True)
        ma_data = []
        for name, val in [("SMA 20", tech["sma_20"]), ("SMA 50", tech["sma_50"]),
                          ("SMA 200", tech["sma_200"]), ("EMA 12", tech["ema_12"]), ("EMA 26", tech["ema_26"])]:
            if val:
                dist = round(((tech["current_price"] - val) / val) * 100, 2)
                ma_data.append({"MA": name, "Value (C$)": usd_to_cad(val),
                                "Position": "Above" if tech["current_price"] > val else "Below",
                                "Distance": f"{dist}%"})
        if ma_data:
            st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Support Levels</p>', unsafe_allow_html=True)
            for lvl in tech["support_levels"]:
                dist = round(((tech["current_price"] - lvl) / lvl) * 100, 2)
                st.write(f"**C${usd_to_cad(lvl)}** — {dist}% below")
            if not tech["support_levels"]:
                st.write("No clear support levels detected")
        with c2:
            st.markdown('<p class="section-header">Resistance Levels</p>', unsafe_allow_html=True)
            for lvl in tech["resistance_levels"]:
                dist = round(((lvl - tech["current_price"]) / tech["current_price"]) * 100, 2)
                st.write(f"**C${usd_to_cad(lvl)}** — {dist}% above")
            if not tech["resistance_levels"]:
                st.write("No clear resistance levels detected")

        st.markdown('<p class="section-header">Bollinger Bands</p>', unsafe_allow_html=True)
        bb_cols = st.columns(4)
        bb_cols[0].metric("Upper", f"C${usd_to_cad(tech['bb_upper'])}")
        bb_cols[1].metric("Middle", f"C${usd_to_cad(tech['bb_middle'])}")
        bb_cols[2].metric("Lower", f"C${usd_to_cad(tech['bb_lower'])}")
        bb_cols[3].metric("Position", f"{tech['bb_position']:.1%}")

    with tab_news:
        if not sent["has_news"]:
            st.warning("No recent news found. Sentiment defaults to neutral.")
        else:
            nc1, nc2, nc3, nc4 = st.columns(4)
            nc1.metric("Bullish", sent["bullish_count"])
            nc2.metric("Neutral", sent["neutral_count"])
            nc3.metric("Bearish", sent["bearish_count"])
            strength_color = "#00c853" if sent["strength"] == "strong" else "#ffd600" if sent["strength"] == "moderate" else "#757575"
            nc4.markdown(f"<div style='background:{strength_color};color:#000;padding:12px;border-radius:8px;text-align:center'>"
                         f"<b>{sent['strength'].title()}</b><br>Signal Strength</div>", unsafe_allow_html=True)

            st.markdown(f"> {sent['significance']}")

            fig_pie = go.Figure(data=[go.Pie(
                labels=["Bullish", "Neutral", "Bearish"],
                values=[sent["bullish_count"], sent["neutral_count"], sent["bearish_count"]],
                marker=dict(colors=["#00c853", "#757575", "#ff5252"]), hole=0.4)])
            fig_pie.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True, key=_next_key("pie"))

            st.markdown('<p class="section-header">Headlines</p>', unsafe_allow_html=True)
            for article in sent["articles"]:
                icon = {
                    "bullish": "\U0001f7e2", "bearish": "\U0001f534", "neutral": "⚪"
                }.get(article["sentiment"], "⚪")
                score_color = "#00c853" if article["compound_score"] > 0.2 else "#ff5252" if article["compound_score"] < -0.2 else "#ffd600"
                st.markdown(f"{icon} **{article['title']}** — _{article['publisher']}_ "
                            f"<span style='color:{score_color}'>(score: {article['compound_score']})</span>",
                            unsafe_allow_html=True)

    with tab_rating:
        st.markdown(
            f"<div style='background:{rating_color};color:#000;padding:24px;border-radius:14px;"
            f"text-align:center;margin-bottom:20px;'>"
            f"<div style='font-size:2em;font-weight:800;margin:0'>{rating['rating']}{arrow}</div>"
            f"<div style='font-size:1.2em;font-weight:600;opacity:0.8;margin-top:4px'>"
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
        fig_bar.update_layout(template="plotly_dark", height=350,
                              xaxis=dict(range=[0, 105], title="Score (0-100)"),
                              margin=dict(l=140, r=50, t=20, b=40))
        st.plotly_chart(fig_bar, use_container_width=True, key=_next_key("bar"))

        st.markdown('<p class="section-header">Reasoning</p>', unsafe_allow_html=True)
        for name, data in rating["component_scores"].items():
            weight_pct = int(data["weight"] * 100)
            score_color = "#00c853" if data["score"] >= 60 else "#ff5252" if data["score"] <= 40 else "#ffd600"
            st.markdown(
                f"<span style='color:{score_color};font-weight:700'>{name.replace('_', ' ').title()}</span> "
                f"<span style='opacity:0.5'>({weight_pct}%)</span> — {data['reasoning']}",
                unsafe_allow_html=True)

        st.markdown('<p class="section-header">Key Signals</p>', unsafe_allow_html=True)
        for signal in rating["key_signals"]:
            icon = "⚡" if "DIVERGENCE" in signal else "→"
            st.write(f"{icon} {signal}")

        st.caption("Mechanical computation — not financial advice.")

    with tab_catalysts:
        render_catalysts(result)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### \U0001f4ca Swing Trader")
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

# --- Main ---
st.caption("⚠️ This tool is for informational and educational purposes only. "
           "Not financial advice. Always do your own research.")

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
        st.markdown("#### Enter a ticker in the sidebar and click Analyze")
        st.markdown("You'll get technicals, sentiment, a 9-component rating, catalysts, "
                    "and a 5-day rating history with day-by-day explanations.")
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**\U0001f527 Technicals**\n\nRSI, MACD, Bollinger Bands, moving averages, support & resistance")
        c2.markdown("**\U0001f4f0 Sentiment**\n\nNews headlines scored for bullish/bearish bias with strength rating")
        c3.markdown("**\U0001f680 Catalysts**\n\nAnalyst targets, insider activity, earnings proximity, relative strength")


# === WATCHLIST ===
with tab_watchlist:
    st.subheader("\U0001f4cb My Watchlist")
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
            st.markdown(f"**Tracking {len(wl_tickers)} tickers:**")
            tag_cols = st.columns(min(len(wl_tickers), 10))
            for i, t in enumerate(wl_tickers):
                with tag_cols[i % min(len(wl_tickers), 10)]:
                    if st.button(f"✕ {t}", key=f"wl_rm_{t}"):
                        remove_from_watchlist(t)
                        st.rerun()

    if not wl_tickers:
        st.markdown("---")
        st.markdown("#### Get started")
        st.markdown("Add tickers above to build your watchlist. Click **Analyze All** "
                    "for a full breakdown sorted bullish to bearish, with sentiment explanations.")
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

            # Sort: most bullish (highest scores) at top, descending to bearish
            valid_results.sort(key=lambda r: r["rating"]["combined_score"], reverse=True)

            # Watchlist overview metrics
            if valid_results:
                avg_score = sum(r["rating"]["combined_score"] for r in valid_results) / len(valid_results)
                bulls = sum(1 for r in valid_results if r["rating"]["rating"] in ("Buy", "Strong Buy"))
                bears = sum(1 for r in valid_results if r["rating"]["rating"] in ("Sell", "Strong Sell"))
                neutrals = len(valid_results) - bulls - bears
                aligned_count = sum(1 for r in valid_results if r.get("sentiment_technical_agreement") == "aligned")

                oc1, oc2, oc3, oc4, oc5 = st.columns(5)
                oc1.metric("Avg Score", f"{avg_score:.0f}/100")
                oc2.metric("Bullish", bulls)
                oc3.metric("Neutral", neutrals)
                oc4.metric("Bearish", bears)
                oc5.metric("High Conviction", aligned_count)

            st.markdown("")

            rows = []
            for r in valid_results:
                tech = r["technicals"]
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, r["rating"]["combined_score"])
                trail = score_trail(hist, r["rating"]["combined_score"])
                agr = r.get("sentiment_technical_agreement", "neutral")
                agr_icon = "\U0001f7e2" if agr == "aligned" else "\U0001f534" if agr == "divergent" else "⚪"

                # Momentum direction from 5-day history
                if len(hist) >= 2:
                    recent_delta = r["rating"]["combined_score"] - hist[0]["score"]
                    if recent_delta >= 8:
                        momentum = "↑ Gaining"
                    elif recent_delta <= -8:
                        momentum = "↓ Fading"
                    else:
                        momentum = "→ Steady"
                else:
                    momentum = "—"

                rows.append({
                    "Ticker": r["ticker"],
                    "Price (C$)": f"{usd_to_cad(tech['current_price']):.2f}",
                    "Change": f"{'+' if tech['price_change_pct'] >= 0 else ''}{tech['price_change_pct']}%",
                    "Rating": f"{r['rating']['rating']}{arr}",
                    "Score": r["rating"]["combined_score"],
                    "Momentum": momentum,
                    "RSI": tech["rsi"],
                    "Trend": tech["trend"].title(),
                    "Sentiment": f"{r['sentiment']['overall_sentiment'].title()}",
                    "Agree": agr_icon,
                })
            if rows:
                st.markdown('<p class="section-header">Overview — Bullish at top, Bearish at bottom</p>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown('<p class="section-header">Sentiment Breakdown</p>', unsafe_allow_html=True)
            for r in valid_results:
                sent = r["sentiment"]
                sent_label = sent["overall_sentiment"].title()
                sent_color = "#00c853" if sent["overall_sentiment"] == "bullish" else "#ff5252" if sent["overall_sentiment"] == "bearish" else "#ffd600"
                strength = sent.get("strength", "none")
                significance = sent.get("significance", "")
                agr = r.get("sentiment_technical_agreement", "neutral")
                agr_text = "Sentiment and technicals agree" if agr == "aligned" else "Sentiment and technicals disagree" if agr == "divergent" else "Mixed signals"

                st.markdown(
                    f"**{r['ticker']}** — "
                    f"<span style='color:{sent_color};font-weight:bold'>{sent_label}</span> "
                    f"({strength}) | {agr_text}",
                    unsafe_allow_html=True)
                st.caption(f"{significance}")

            csv_rows = list(rows)
            if csv_rows:
                csv = pd.DataFrame(csv_rows).to_csv(index=False)
                st.download_button("Download CSV", csv, "watchlist_analysis.csv", "text/csv")

            # Per-ticker cards: sorted bull→bear (highest score at top)
            st.divider()
            for r in valid_results:
                if r.get("error"):
                    continue
                tech = r["technicals"]
                sent = r["sentiment"]
                rating = r["rating"]
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, rating["combined_score"])
                rc = RATING_COLORS.get(rating["rating"], "#757575")
                price_color = "#00c853" if tech["price_change_pct"] >= 0 else "#ff5252"
                agr = r.get("sentiment_technical_agreement", "neutral")

                # Determine "what am I missing" insight for this ticker
                insights = []
                if tech["rsi"] < 30 and rating["combined_score"] < 50:
                    insights.append("\U0001f4a1 RSI oversold but rating is bearish — possible value trap or bottom-fishing opportunity. Wait for MACD confirmation.")
                elif tech["rsi"] > 70 and rating["combined_score"] > 65:
                    insights.append("\U0001f4a1 RSI overbought while rating is bullish — momentum is strong but entry risk is elevated. Consider waiting for a pullback.")
                if agr == "divergent":
                    insights.append("\U0001f4a1 Sentiment and technicals disagree — dig deeper. Is the market pricing in something the charts haven't shown yet?")
                earnings = r.get("earnings", {})
                if earnings.get("has_date") and earnings.get("in_swing_window"):
                    eps_hint = f" EPS est: ${earnings['eps_estimate']:.2f}." if earnings.get("eps_estimate") is not None else ""
                    track_hint = ""
                    if earnings.get("history"):
                        track_hint = f" Last {len(earnings['history'])}Q: {earnings['beat_count']} beat, {earnings['miss_count']} miss (avg surprise {earnings.get('avg_surprise_pct', 0):+.1f}%)."
                    insights.append(f"\U0001f4a1 Earnings in {earnings['days_until']} days — this changes the risk profile.{eps_hint}{track_hint} Decide before the event, not during it.")
                if tech["support_levels"] and tech["current_price"] > 0:
                    nearest_sup = tech["support_levels"][0]
                    sup_dist = ((tech["current_price"] - nearest_sup) / tech["current_price"]) * 100
                    if sup_dist < 2:
                        insights.append("\U0001f4a1 Price is very close to support — a bounce here could be a strong entry, but a break below means exit fast.")
                if tech["resistance_levels"] and tech["current_price"] > 0:
                    nearest_res = tech["resistance_levels"][0]
                    res_dist = ((nearest_res - tech["current_price"]) / tech["current_price"]) * 100
                    if res_dist < 2:
                        insights.append("\U0001f4a1 Price pressing against resistance — a breakout above could accelerate gains, but failure here means rejection.")

                with st.expander(f"{r['ticker']} — {r['info']['shortName']} | {rating['rating']}{arr} ({rating['combined_score']:.0f}/100)"):
                    wc1, wc2, wc3 = st.columns([2.5, 1, 1])
                    with wc1:
                        st.markdown(f"#### {r['ticker']} — {r['info']['shortName']}")
                        st.markdown(
                            f"<span style='font-size:1.3em;font-weight:700'>C${usd_to_cad(tech['current_price']):.2f}</span> "
                            f"<span style='color:{price_color};font-weight:600'>"
                            f"({'+' if tech['price_change_pct'] >= 0 else ''}{tech['price_change_pct']}%)</span>",
                            unsafe_allow_html=True)
                    with wc2:
                        st.metric("RSI", f"{tech['rsi']:.1f}",
                                  "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
                    with wc3:
                        st.markdown(
                            f"<div class='rating-badge' style='background:{rc}'>"
                            f"<div class='label'>{rating['rating']}{arr}</div>"
                            f"<div class='score'>{rating['combined_score']:.0f}/100</div></div>",
                            unsafe_allow_html=True)

                    qm1, qm2, qm3, qm4 = st.columns(4)
                    qm1.metric("Trend", tech['trend'].title())
                    qm2.metric("Vol Ratio", f"{tech['vol_ratio']}x",
                               "Above avg" if tech["vol_ratio"] > 1.2 else "Below avg" if tech["vol_ratio"] < 0.8 else "Avg")
                    trail = score_trail(r.get("rating_history", []), rating["combined_score"])
                    qm3.metric("5-Day", trail.split(" → ")[-1] if trail else "—",
                               f"from {trail.split(' → ')[0]}" if " → " in trail else "")
                    qm4.metric("MACD", f"{tech['macd_hist']:.4f}",
                               "Expanding" if abs(tech["macd_hist"]) > abs(tech["macd_hist_prev"]) else "Contracting")

                    st.markdown("---")
                    st.markdown('<p class="section-header">Sentiment</p>', unsafe_allow_html=True)
                    render_position_sentiment(r)

                    if insights:
                        st.markdown("---")
                        st.markdown('<p class="section-header">What You Might Be Missing</p>', unsafe_allow_html=True)
                        for ins in insights:
                            st.markdown(ins)

                    st.markdown("---")
                    st.markdown('<p class="section-header">Key Signals</p>', unsafe_allow_html=True)
                    for signal in rating["key_signals"]:
                        icon = "⚡" if "DIVERGENCE" in signal else "→"
                        st.write(f"{icon} {signal}")

                    render_rating_history(r)

                    with st.expander("View Full Analysis", expanded=False):
                        render_analysis(r)

            if error_results:
                st.divider()
                st.markdown("##### Errors")
                for r in error_results:
                    st.error(f"**{r['ticker']}:** {r['error']}")


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
        pos_sl = fc5.number_input("Stop Loss (USD)", min_value=0.0, step=0.01, format="%.2f", value=0.0,
                                   help="Optional — 0 to skip")
        pos_tp = fc6.number_input("Target Price (USD)", min_value=0.0, step=0.01, format="%.2f", value=0.0,
                                   help="Optional — 0 to skip")
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

        # Portfolio summary
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Cost Basis", f"C${usd_to_cad(total_cost):,.2f}")
        sc2.metric("Market Value", f"C${usd_to_cad(total_val):,.2f}")
        pfx = "+" if total_pnl >= 0 else ""
        sc3.metric("Total P&L", f"{pfx}C${usd_to_cad(total_pnl):,.2f}", f"{pfx}{total_pnl_pct:.1f}%")
        sc4.metric("Positions", len(positions))

        # Summary table
        summary = []
        for pos, a, g in pos_results:
            pnl_s = f"{'+' if g['pnl_pct'] >= 0 else ''}{g['pnl_pct']}%"
            hist = a.get("rating_history", []) if not a.get("error") else []
            score_now = a["rating"]["combined_score"] if not a.get("error") else 0
            arr = trend_arrow(hist, score_now)
            summary.append({
                "Ticker": pos["ticker"],
                "Entry (C$)": f"{usd_to_cad(pos['entry_price']):.2f}",
                "Now (C$)": f"{usd_to_cad(g['current_price']):.2f}",
                "Shares": pos["shares"],
                "P&L": f"{'+' if g['pnl_total'] >= 0 else ''}{usd_to_cad(g['pnl_total']):.2f} ({pnl_s})",
                "Days": g["days_held"],
                "Rating": f"{a['rating']['rating']}{arr}" if not a.get("error") else "N/A",
                "Action": g["action"],
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.divider()

        # Individual position cards
        for pos, a, g in pos_results:
            ac_color = ACTION_COLORS.get(g["action"], "#757575")
            label = f"{pos['ticker']} — {g['action']} ({'+' if g['pnl_pct'] >= 0 else ''}{g['pnl_pct']}%)"
            with st.expander(label, expanded=(g["urgency"] == "high")):
                # Position header
                phc1, phc2, phc3 = st.columns([2, 1, 1])
                with phc1:
                    st.markdown(f"**Entry:** C${usd_to_cad(pos['entry_price']):.2f} on {pos['entry_date']} ({pos['shares']} shares)")
                    if pos.get("stop_loss"):
                        st.markdown(f"**Stop Loss:** C${usd_to_cad(pos['stop_loss']):.2f}")
                    elif g.get("atr_suggested_stop"):
                        st.markdown(f"**Suggested Stop (2×ATR):** C${usd_to_cad(g['atr_suggested_stop']):.2f}")
                    if pos.get("target_price"):
                        st.markdown(f"**Target:** C${usd_to_cad(pos['target_price']):.2f}")
                    elif g.get("atr_suggested_target"):
                        st.markdown(f"**Suggested Target (3×ATR):** C${usd_to_cad(g['atr_suggested_target']):.2f}")
                    if g.get("atr_trailing_stop"):
                        st.markdown(f"**Trailing Stop (1.5×ATR):** C${usd_to_cad(g['atr_trailing_stop']):.2f}")
                    if pos.get("notes"):
                        st.markdown(f"*{pos['notes']}*")
                with phc2:
                    pfx = "+" if g["pnl_pct"] >= 0 else ""
                    st.metric("P&L", f"{pfx}C${usd_to_cad(g['pnl_total']):.2f}", f"{pfx}{g['pnl_pct']:.1f}%")
                with phc3:
                    st.markdown(
                        f"<div class='rating-badge' style='background:{ac_color}'>"
                        f"<div class='label'>{g['action']}</div></div>",
                        unsafe_allow_html=True)

                # Metrics row
                dm1, dm2, dm3, dm4, dm5 = st.columns(5)
                if not a.get("error"):
                    tech = a["technicals"]
                    dm1.metric("RSI", f"{tech['rsi']:.1f}",
                               "Oversold" if tech["rsi"] < 30 else "Overbought" if tech["rsi"] > 70 else "Normal")
                    dm2.metric("Trend", tech["trend"].title())
                    dm3.metric("Rating", a["rating"]["rating"], f"Score: {a['rating']['combined_score']:.0f}")
                sw = 15
                dr = max(0, sw - g["trading_days_held"])
                dm4.metric("Days", f"{g['days_held']}d", f"{dr} trading days left" if dr > 0 else "Past window")
                if pos.get("stop_loss") and pos.get("target_price"):
                    risk = abs(pos["entry_price"] - pos["stop_loss"])
                    reward = abs(pos["target_price"] - pos["entry_price"])
                    rr = round(reward / risk, 2) if risk > 0 else 0
                    dm5.metric("R/R", f"{rr}:1", "Good" if rr >= 2 else "Fair" if rr >= 1 else "Poor")
                else:
                    dm5.metric("R/R", "N/A", "Set stop & target")

                # Current Sentiment for this position
                if not a.get("error"):
                    st.markdown("---")
                    st.markdown('<p class="section-header">Current Sentiment</p>', unsafe_allow_html=True)
                    render_position_sentiment(a)

                    # Sentiment shift since entry
                    sent = a.get("sentiment", {})
                    entry_sent = pos.get("entry_sentiment_score")
                    if entry_sent is not None and sent.get("has_news"):
                        now_sent = sent["sentiment_score"]
                        shift = now_sent - entry_sent
                        if abs(shift) > 5:
                            shift_color = "#00c853" if shift > 0 else "#ff5252"
                            shift_dir = "improved" if shift > 0 else "deteriorated"
                            st.markdown(f"**Sentiment Shift Since Entry:** "
                                        f"<span style='color:{shift_color};font-weight:bold'>"
                                        f"{entry_sent:.0f} → {now_sent:.0f} ({shift:+.0f}) — "
                                        f"Sentiment has {shift_dir}</span>",
                                        unsafe_allow_html=True)

                if not a.get("error"):
                    tech = a["technicals"]
                    st.markdown("---")
                    lv1, lv2 = st.columns(2)
                    with lv1:
                        if tech["support_levels"]:
                            ns = tech["support_levels"][0]
                            sd = round(((tech["current_price"] - ns) / tech["current_price"]) * 100, 1)
                            st.markdown(f"**Nearest Support:** C${usd_to_cad(ns)} ({sd}% below)")
                        else:
                            st.markdown("**Nearest Support:** none detected")
                    with lv2:
                        if tech["resistance_levels"]:
                            nr = tech["resistance_levels"][0]
                            rd = round(((nr - tech["current_price"]) / tech["current_price"]) * 100, 1)
                            st.markdown(f"**Nearest Resistance:** C${usd_to_cad(nr)} ({rd}% above)")
                        else:
                            st.markdown("**Nearest Resistance:** none detected")

                    earnings = a.get("earnings", {})
                    if earnings.get("has_date") and earnings.get("in_swing_window"):
                        eps_str = f" | EPS Est: ${earnings['eps_estimate']:.2f}" if earnings.get("eps_estimate") is not None else ""
                        track = ""
                        if earnings.get("history"):
                            track = f" | Track: {earnings['beat_count']} beat / {earnings['miss_count']} miss (avg {earnings.get('avg_surprise_pct', 0):+.1f}%)"
                        st.warning(f"⚠️ Earnings in {earnings['days_until']} days ({earnings['date_str']}){eps_str}{track}")

                    render_rating_history(a)

                st.markdown("---")
                st.markdown('<p class="section-header">Guidance</p>', unsafe_allow_html=True)
                for reason in g["reasons"]:
                    icon = "→"
                    if "STOP LOSS" in reason or "TARGET REACHED" in reason or "EARNINGS" in reason:
                        icon = "\U0001f6a8"
                    elif "bullish" in reason.lower() or "support" in reason.lower() or "good" in reason.lower():
                        icon = "\U0001f7e2"
                    elif "bearish" in reason.lower() or "overbought" in reason.lower() or "loss" in reason.lower() or "deteriorat" in reason.lower():
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
                    mini.update_layout(template="plotly_dark", height=250,
                                       margin=dict(l=50, r=50, t=10, b=30),
                                       yaxis_title="Price (C$)", showlegend=False)
                    st.plotly_chart(mini, use_container_width=True, key=_next_key("pos_mini"))

                st.markdown("---")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    with st.form(f"edit_{pos['id']}"):
                        st.markdown("**Edit Position**")
                        new_sl = st.number_input("Stop Loss (USD)", value=pos.get("stop_loss") or 0.0,
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
                        if st.form_submit_button("Close & Save to History"):
                            close_position_with_history(pos["id"], exit_price)
                            st.rerun()
                with ec3:
                    st.markdown("**Remove**")
                    st.caption("Removes without saving to history")
                    if st.button("Remove Position", key=f"del_{pos['id']}"):
                        remove_position(pos["id"])
                        st.rerun()


# === MARKET MOVERS ===
with tab_movers:
    st.markdown("#### \U0001f525 Market Movers")
    st.caption("Scans 45+ stocks for high-impact headlines (sentiment ≥ 0.4). Only strong directional stories.")

    if st.button("\U0001f525 Scan for Market Movers", type="primary", use_container_width=True):
        with st.spinner("Scanning headlines across all tracked stocks..."):
            movers = cached_market_movers()

        if not movers:
            st.info("No high-impact stories found right now. Markets may be quiet or news hasn't broken yet.")
        else:
            bull_movers = [m for m in movers if m["sentiment"] == "bullish"]
            bear_movers = [m for m in movers if m["sentiment"] == "bearish"]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Big Stories", len(movers))
            mc2.metric("Bullish", len(bull_movers))
            mc3.metric("Bearish", len(bear_movers))

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown('<p class="section-header">\U0001f7e2 Bullish Movers</p>', unsafe_allow_html=True)
                if bull_movers:
                    for m in bull_movers[:10]:
                        score_pct = int(m["compound"] * 100)
                        st.markdown(
                            f"**{m['ticker']}** — {m['title'][:70]} "
                            f"<span style='color:#00c853;font-weight:bold'>(+{score_pct}%)</span> "
                            f"<span style='opacity:0.5'>— {m['publisher']}</span>",
                            unsafe_allow_html=True)
                else:
                    st.caption("No bullish movers found.")

            with bc2:
                st.markdown('<p class="section-header">\U0001f534 Bearish Movers</p>', unsafe_allow_html=True)
                if bear_movers:
                    for m in bear_movers[:10]:
                        score_pct = int(abs(m["compound"]) * 100)
                        st.markdown(
                            f"**{m['ticker']}** — {m['title'][:70]} "
                            f"<span style='color:#ff5252;font-weight:bold'>(-{score_pct}%)</span> "
                            f"<span style='opacity:0.5'>— {m['publisher']}</span>",
                            unsafe_allow_html=True)
                else:
                    st.caption("No bearish movers found.")

            st.divider()
            mover_rows = []
            for m in movers:
                mover_rows.append({
                    "Ticker": m["ticker"],
                    "Headline": m["title"][:80],
                    "Sentiment": m["sentiment"].title(),
                    "Score": m["compound"],
                    "Source": m["publisher"],
                })
            st.dataframe(pd.DataFrame(mover_rows), hide_index=True, use_container_width=True)

            csv = pd.DataFrame(mover_rows).to_csv(index=False)
            st.download_button("Download Movers CSV", csv, "market_movers.csv", "text/csv")
    else:
        st.markdown("")
        st.markdown("Hit the button above to scan 45+ stocks for the biggest sentiment-moving headlines. "
                    "Only stories with strong directional sentiment (score >= 0.4) are shown.")
        st.markdown("")
        c1, c2 = st.columns(2)
        c1.markdown("**\U0001f7e2 Bullish Movers**\n\nStocks with strongly positive headline sentiment")
        c2.markdown("**\U0001f534 Bearish Movers**\n\nStocks with strongly negative headline sentiment")


# === SCREENER ===
with tab_screener:
    st.markdown("#### \U0001f4e1 Screener")
    fc1, fc2, fc3, fc4 = st.columns(4)
    min_score = fc1.slider("Min Score", 0, 100, 0)
    rsi_filter = fc2.selectbox("RSI Zone", ["All", "Oversold (<35)", "Overbought (>65)"])
    trend_filter = fc3.selectbox("Trend", ["All", "Bullish", "Bearish"])
    sent_filter = fc4.selectbox("Sentiment", ["All", "Bullish", "Bearish", "High Conviction"])
    st.caption("45+ stocks • 5-min cache")

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
            if rsi_filter == "Oversold (<35)" and r["technicals"]["rsi"] >= 35:
                continue
            if rsi_filter == "Overbought (>65)" and r["technicals"]["rsi"] <= 65:
                continue
            if trend_filter == "Bullish" and "bullish" not in r["technicals"]["trend"]:
                continue
            if trend_filter == "Bearish" and "bearish" not in r["technicals"]["trend"]:
                continue
            if sent_filter == "Bullish" and r["sentiment"]["overall_sentiment"] != "bullish":
                continue
            if sent_filter == "Bearish" and r["sentiment"]["overall_sentiment"] != "bearish":
                continue
            if sent_filter == "High Conviction" and r.get("sentiment_technical_agreement") != "aligned":
                continue
            filtered.append(r)

        buys = [r for r in filtered if r["rating"]["rating"] in ("Strong Buy", "Buy")]
        buys.sort(key=lambda r: r["rating"]["combined_score"], reverse=True)
        others = [r for r in filtered if r["rating"]["rating"] not in ("Strong Buy", "Buy")]
        others.sort(key=lambda r: r["rating"]["combined_score"], reverse=True)

        if buys:
            st.subheader(f"Recommended Setups ({len(buys)} found)")
            buy_rows = []
            for r in buys:
                tech = r["technicals"]
                rating = r["rating"]
                sent = r["sentiment"]
                hist = r.get("rating_history", [])
                arr = trend_arrow(hist, rating["combined_score"])
                trail = score_trail(hist, rating["combined_score"])
                agr = r.get("sentiment_technical_agreement", "neutral")
                agr_icon = "\U0001f7e2" if agr == "aligned" else "\U0001f534" if agr == "divergent" else "⚪"
                buy_rows.append({
                    "Ticker": r["ticker"],
                    "Name": r["info"]["shortName"],
                    "Price (C$)": f"{usd_to_cad(tech['current_price']):.2f}",
                    "Rating": f"{rating['rating']}{arr}",
                    "Score": rating["combined_score"],
                    "5-Day Trend": trail,
                    "RSI": tech["rsi"],
                    "Trend": tech["trend"].title(),
                    "Sentiment": f"{sent['overall_sentiment'].title()} ({sent.get('strength', '')})",
                    "Agree": agr_icon,
                })
            st.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)

            csv = pd.DataFrame(buy_rows).to_csv(index=False)
            st.download_button("Download CSV", csv, "screener_buys.csv", "text/csv")

            wl = load_watchlist()
            wl_cols = st.columns(min(len(buys), 6))
            for i, r in enumerate(buys[:6]):
                with wl_cols[i]:
                    if r["ticker"] not in wl:
                        if st.button(f"+ {r['ticker']}", key=f"scr_add_{r['ticker']}"):
                            add_to_watchlist(r["ticker"])
                            st.rerun()

            for r in buys:
                with st.expander(f"{r['ticker']} — {r['rating']['rating']} ({r['rating']['combined_score']:.0f}/100)"):
                    render_analysis(r)
        else:
            st.warning("No Buy or Strong Buy setups match your filters.")

        if others:
            st.divider()
            with st.expander(f"All Other Stocks ({len(others)})"):
                other_rows = []
                for r in others:
                    tech = r["technicals"]
                    hist = r.get("rating_history", [])
                    arr = trend_arrow(hist, r["rating"]["combined_score"])
                    other_rows.append({
                        "Ticker": r["ticker"],
                        "Price (C$)": f"{usd_to_cad(tech['current_price']):.2f}",
                        "Rating": f"{r['rating']['rating']}{arr}",
                        "Score": r["rating"]["combined_score"],
                        "RSI": tech["rsi"],
                        "Trend": tech["trend"].title(),
                    })
                st.dataframe(pd.DataFrame(other_rows), hide_index=True, use_container_width=True)
    else:
        st.markdown("Click **Run Screener** to scan for setups. Use filters above to narrow results.")


# === TRADE HISTORY ===
with tab_history:
    st.markdown("#### \U0001f4ca Trade History")
    stats = compute_trade_statistics()
    if not stats["has_data"]:
        st.markdown("")
        st.markdown("#### No closed trades yet")
        st.markdown("Close a position from the **Positions** tab to start tracking your performance. "
                    "You'll see win rate, profit factor, equity curve, and per-trade P&L charts here.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Win / Loss Split**\n\nWins and losses cataloged separately with P&L details")
        c2.markdown("**Performance Chart**\n\nPer-trade bar chart with average P&L line")
        c3.markdown("**Equity Curve**\n\nCumulative P&L over time with export to CSV")
    else:
        history = stats["history"]

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Trades", stats["total_trades"])
        mc2.metric("Win Rate", f"{stats['win_rate']}%", f"{stats['wins']}W / {stats['losses']}L")
        mc3.metric("Profit Factor", f"{stats['profit_factor']}")
        all_pcts = [t["pnl_pct"] for t in history]
        avg_all = sum(all_pcts) / len(all_pcts) if all_pcts else 0
        mc4.metric("Avg Trade", f"{avg_all:+.2f}%")
        pnl_color = "+" if stats['total_pnl'] >= 0 else ""
        mc5.metric("Total P&L", f"C${usd_to_cad(stats['total_pnl']):,.2f}")

        mc6, mc7, mc8 = st.columns(3)
        mc6.metric("Avg Win", f"{stats['avg_gain']:+.2f}%")
        mc7.metric("Avg Loss", f"{stats['avg_loss']:+.2f}%")
        mc8.metric("Avg Hold", f"{stats['avg_hold_days']} days")

        st.divider()

        # Wins & Losses catalog
        wins = [t for t in history if t["pnl_pct"] >= 0]
        losses = [t for t in history if t["pnl_pct"] < 0]

        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown(f'<p class="section-header">\U0001f7e2 Wins ({len(wins)})</p>', unsafe_allow_html=True)
            if wins:
                win_rows = []
                for t in sorted(wins, key=lambda x: x["pnl_pct"], reverse=True):
                    win_rows.append({
                        "Ticker": t["ticker"],
                        "Entry (C$)": f"{usd_to_cad(t['entry_price']):.2f}",
                        "Exit (C$)": f"{usd_to_cad(t['exit_price']):.2f}",
                        "P&L %": f"+{t['pnl_pct']:.2f}%",
                        "P&L (C$)": f"+C${usd_to_cad(t['pnl_total']):.2f}",
                        "Held": f"{t['days_held']}d",
                    })
                st.dataframe(pd.DataFrame(win_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No winning trades yet.")

        with wc2:
            st.markdown(f'<p class="section-header">\U0001f534 Losses ({len(losses)})</p>', unsafe_allow_html=True)
            if losses:
                loss_rows = []
                for t in sorted(losses, key=lambda x: x["pnl_pct"]):
                    loss_rows.append({
                        "Ticker": t["ticker"],
                        "Entry (C$)": f"{usd_to_cad(t['entry_price']):.2f}",
                        "Exit (C$)": f"{usd_to_cad(t['exit_price']):.2f}",
                        "P&L %": f"{t['pnl_pct']:.2f}%",
                        "P&L (C$)": f"C${usd_to_cad(t['pnl_total']):.2f}",
                        "Held": f"{t['days_held']}d",
                    })
                st.dataframe(pd.DataFrame(loss_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No losing trades yet.")

        st.divider()

        st.markdown('<p class="section-header">Trade Performance</p>', unsafe_allow_html=True)
        trade_labels = [f"{t['ticker']}\n{t['exit_date']}" for t in history]
        trade_pcts = [t["pnl_pct"] for t in history]
        bar_colors = ["#00c853" if p >= 0 else "#ff5252" for p in trade_pcts]

        fig_trades = go.Figure()
        fig_trades.add_trace(go.Bar(
            x=trade_labels, y=trade_pcts,
            marker_color=bar_colors,
            text=[f"{p:+.1f}%" for p in trade_pcts],
            textposition="outside",
        ))
        fig_trades.add_hline(y=avg_all, line_dash="dash", line_color="#42a5f5", line_width=2,
                             annotation_text=f"Avg: {avg_all:+.1f}%")
        fig_trades.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)
        fig_trades.update_layout(
            template="plotly_dark", height=350,
            yaxis_title="P&L %",
            xaxis_title="Trade",
            margin=dict(l=50, r=50, t=20, b=80),
            showlegend=False,
        )
        st.plotly_chart(fig_trades, use_container_width=True, key=_next_key("trades_bar"))

        st.markdown('<p class="section-header">Equity Curve</p>', unsafe_allow_html=True)
        cum_pnl = []
        running = 0
        for t in history:
            running += t["pnl_total"]
            cum_pnl.append(running)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=[t["exit_date"] for t in history], y=[usd_to_cad(p) for p in cum_pnl],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#42a5f5", width=2), fillcolor="rgba(66,165,245,0.1)"))
        fig_eq.update_layout(template="plotly_dark", height=300,
                             yaxis_title="Cumulative P&L (C$)",
                             margin=dict(l=50, r=50, t=20, b=30))
        st.plotly_chart(fig_eq, use_container_width=True, key=_next_key("equity_curve"))

        st.markdown('<p class="section-header">All Trades</p>', unsafe_allow_html=True)
        hist_rows = []
        for t in history:
            hist_rows.append({
                "Ticker": t["ticker"],
                "Entry (C$)": f"{usd_to_cad(t['entry_price']):.2f}",
                "Exit (C$)": f"{usd_to_cad(t['exit_price']):.2f}",
                "P&L %": f"{t['pnl_pct']:+.2f}%",
                "P&L (C$)": f"C${usd_to_cad(t['pnl_total']):,.2f}",
                "Held": f"{t['days_held']}d",
                "Entry Date": t["entry_date"],
                "Exit Date": t["exit_date"],
            })
        st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)

        csv = pd.DataFrame(hist_rows).to_csv(index=False)
        st.download_button("Download Trade History", csv, "trade_history.csv", "text/csv")
