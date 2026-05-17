"""
monte carlo stock price simulation - streamlit app
====================================================
all data derived from yfinance and cached for performance
run with:
    streamlit run mcarlo_app.py

features:
  - preset configurations (Conservative / Balanced / Aggressive / Momentum)
  - interactive Plotly charts (zoom, pan, hover crosshair)
  - downloadable chart (HTML) and results (CSV)
  - VaR & CVaR risk metrics panel
  - price target probability tool
  - portfolio mode (multi-asset Monte Carlo)
  - AI-powered plain-English summary via Groq (Llama 3.3 70B)
  - about / methodology page
  - EWMA volatility, Student-t fat tails, recent-window drift, no fixed seed
"""

import warnings
warnings.filterwarnings("ignore")

import io
import csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st
from datetime import date, datetime, timedelta
from scipy.stats import t as student_t
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Price Action Forecast via Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
)

# ── Shared Plotly theme ───────────────────────────────────────────────────────

THEME = dict(
    bg        = "#0f0f0f",
    panel     = "#1a1a1a",
    text      = "#e0e0e0",
    muted     = "#777777",
    grid      = "#2e2e2e",
    blue      = "#378ADD",
    green     = "#639922",
    red       = "#D85A30",
    amber     = "#BA7517",
    hist_col  = "#1c3a5e",
    fore_col  = "#4caf78",
)

PORTFOLIO_COLORS = [
    "#378ADD", "#4caf78", "#BA7517", "#D85A30",
    "#9b59b6", "#1abc9c", "#e74c3c", "#f39c12",
]

def base_layout(title="", xaxis=None, yaxis=None, height=420):
    layout = dict(
        title         = dict(text=title, font=dict(color=THEME["text"], size=13)),
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["panel"],
        font          = dict(color=THEME["muted"], size=11),
        legend        = dict(
            bgcolor     = "rgba(26,26,26,0.8)",
            bordercolor = "#444",
            borderwidth = 1,
            font        = dict(color=THEME["text"], size=10),
        ),
        hovermode = "x unified",
        height    = height,
        margin    = dict(l=60, r=30, t=60, b=50),
        xaxis = dict(
            gridcolor     = THEME["grid"],
            zerolinecolor = THEME["grid"],
            tickfont      = dict(color=THEME["muted"]),
            **(xaxis or {}),
        ),
        yaxis = dict(
            gridcolor     = THEME["grid"],
            zerolinecolor = THEME["grid"],
            tickfont      = dict(color=THEME["muted"]),
            **(yaxis or {}),
        ),
    )
    return layout

# ── Preset definitions ────────────────────────────────────────────────────────

PRESETS = {
    "Custom": None,
    "Conservative": {
        "vol_scale":    0.75,
        "ewma_lambda":  0.97,
        "t_dof":        15,
        "drift_window": 126,
        "description":  "Lower volatility, slow EWMA decay, near-normal tails, long drift window. "
                        "Best for large-cap, low-beta stocks (e.g. JNJ, KO, BRK).",
    },
    "Balanced": {
        "vol_scale":    1.0,
        "ewma_lambda":  0.94,
        "t_dof":        5,
        "drift_window": 63,
        "description":  "RiskMetrics defaults with fat tails and a 3-month drift window. "
                        "A solid starting point for most equities.",
    },
    "Aggressive": {
        "vol_scale":    1.5,
        "ewma_lambda":  0.90,
        "t_dof":        3,
        "drift_window": 42,
        "description":  "Amplified volatility, fast EWMA decay, very fat tails. "
                        "Suitable for high-beta or speculative names (e.g. TSLA, MSTR, MEME stocks).",
    },
    "Momentum": {
        "vol_scale":    1.0,
        "ewma_lambda":  0.92,
        "t_dof":        5,
        "drift_window": 21,
        "description":  "Standard vol but extremely short drift window (1 month). "
                        "Captures near-term momentum; useful when a stock is in a strong trend.",
    },
}

# ── Shared core functions (used by both Simulator and Portfolio tabs) ─────────

def add_trading_days(from_date: date, n_days: int) -> list:
    dates, cursor = [], from_date
    while len(dates) < n_days:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            dates.append(cursor)
    return dates

@st.cache_data(show_spinner="Fetching historical data from yfinance…")
def fetch_data(ticker: str, start: date, end: date):
    end_inclusive = end + timedelta(days=1)
    df = yf.download(
        ticker,
        start=str(start),
        end=str(end_inclusive),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(
            f"No data found for **{ticker}** in the range {start} → {end}. "
            "Check the ticker symbol and date range."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    closes = df["Close"].dropna()
    if len(closes) < 20:
        raise ValueError(
            f"Only {len(closes)} trading days in range — need at least 20."
        )
    info = {}
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        pass
    return closes, info

@st.cache_data(show_spinner=False)
def fetch_actual_data(ticker: str, from_date: date):
    try:
        end_inclusive = date.today() + timedelta(days=1)
        df = yf.download(
            ticker,
            start=str(from_date),
            end=str(end_inclusive),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        closes = df["Close"].dropna()
        return closes if len(closes) > 1 else None
    except Exception:
        return None

def compute_ewma_vol(log_returns: pd.Series, lam: float) -> float:
    squared  = log_returns.values ** 2
    ewma_var = squared[0]
    for r2 in squared[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * r2
    return float(np.sqrt(ewma_var))

def compute_parameters(closes, drift_override=None, vol_scale=1.0,
                       ewma_lambda=0.94, drift_window=63):
    log_returns      = np.log(closes / closes.shift(1)).dropna()
    ann_drift_full   = log_returns.mean() * 252
    ann_vol_simple   = log_returns.std()  * np.sqrt(252)
    daily_ewma_vol   = compute_ewma_vol(log_returns, ewma_lambda)
    ann_vol_ewma     = daily_ewma_vol * np.sqrt(252)
    recent_returns   = log_returns.iloc[-drift_window:] if len(log_returns) >= drift_window else log_returns
    daily_drift      = recent_returns.mean()
    ann_drift_recent = daily_drift * 252
    if drift_override is not None:
        daily_drift = drift_override / 252
    daily_std_scaled = daily_ewma_vol * vol_scale
    return {
        "daily_mean":        daily_drift,
        "daily_std":         daily_std_scaled,
        "ann_drift_full":    ann_drift_full,
        "ann_drift_recent":  ann_drift_recent,
        "ann_vol_simple":    ann_vol_simple,
        "ann_vol_ewma":      ann_vol_ewma,
        "ann_vol_scaled":    daily_std_scaled * np.sqrt(252),
        "log_returns":       log_returns,
        "last_price":        float(closes.iloc[-1]),
        "num_hist_days":     len(closes),
        "drift_window_used": min(drift_window, len(log_returns)),
    }

def run_simulation(params, days, n_sims, t_dof):
    mu, sigma, S0 = params["daily_mean"], params["daily_std"], params["last_price"]
    raw           = student_t.rvs(df=t_dof, size=(n_sims, days))
    scale_factor  = np.sqrt(t_dof / (t_dof - 2))
    Z             = raw / scale_factor
    daily_returns = (mu - 0.5 * sigma**2) + sigma * Z
    log_paths     = np.concatenate([np.zeros((n_sims, 1)), daily_returns], axis=1)
    return S0 * np.exp(np.cumsum(log_paths, axis=1))

def compute_percentiles(paths):
    return {
        "p5":  np.percentile(paths,  5, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }

def find_median_path(paths, pcts):
    closest_idx = np.argmin(np.abs(paths[:, -1] - pcts["p50"][-1]))
    return paths[closest_idx]

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_sim, tab_port, tab_about = st.tabs([
    "📈 Simulator",
    "💼 Portfolio Mode",
    "📖 About & Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sim:

    st.title("📈 Price Action Forecast via Monte Carlo Simulation")
    st.markdown(
        "Fetches real historical data via **yfinance** and simulates future price "
        "paths using **Geometric Brownian Motion (GBM)** with EWMA volatility and "
        "fat-tailed returns."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.header("⚙️ Simulation Parameters")

        ticker = st.text_input(
            "Ticker symbol",
            value="AAPL",
            help="e.g. AAPL, TSLA, NVDA, SPY",
        ).strip().upper()

        st.subheader("Historical data window")
        today  = date.today()
        one_yr = today - timedelta(days=365)

        start_date = st.date_input(
            "Start date",
            value=one_yr,
            max_value=today - timedelta(days=31),
            help="Must be at least 30 days before end date.",
        )
        end_date = st.date_input(
            "End date",
            value=today,
            min_value=start_date + timedelta(days=30),
            max_value=today,
        )

        st.subheader("Forecast settings")
        days_forward = st.slider(
            "Trading days to simulate forward",
            min_value=21, max_value=756, value=252, step=21,
            help="252 ≈ 1 year of trading days.",
        )
        num_simulations = st.slider(
            "Number of simulations",
            min_value=100, max_value=10000, value=500, step=100,
        )

        st.subheader("Model preset")
        preset_name = st.selectbox(
            "Choose a preset",
            options=list(PRESETS.keys()),
            index=2,
            help="Presets auto-fill the Advanced sliders. Switch to Custom to tweak freely.",
        )
        preset = PRESETS[preset_name]
        if preset:
            st.caption(f"ℹ️ {preset['description']}")

        st.subheader("Advanced")

        def _val(key, default):
            return preset[key] if preset else default

        vol_scale = st.slider(
            "Volatility multiplier",
            min_value=0.25, max_value=3.0, step=0.25,
            value=_val("vol_scale", 1.0),
        )
        ewma_lambda = st.slider(
            "EWMA decay factor (λ)",
            min_value=0.80, max_value=0.99, step=0.01,
            value=_val("ewma_lambda", 0.94),
        )
        t_dof = st.slider(
            "Student-t degrees of freedom",
            min_value=3, max_value=30, step=1,
            value=_val("t_dof", 5),
        )
        drift_window = st.slider(
            "Recent drift window (trading days)",
            min_value=21, max_value=252, step=21,
            value=_val("drift_window", 63),
        )

        use_drift_override = st.checkbox("Override drift manually?", value=False)
        drift_override = None
        if use_drift_override:
            drift_override = st.number_input(
                "Annual drift (e.g. 0.10 = +10%)",
                min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f",
            )

        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    # ── VaR & CVaR ───────────────────────────────────────────────────────────

    def compute_risk_metrics(paths, S0, confidence_levels=(0.90, 0.95, 0.99)):
        final_prices = paths[:, -1]
        pnl          = final_prices - S0
        pnl_pct      = (final_prices / S0 - 1) * 100
        rows = []
        for cl in confidence_levels:
            alpha     = 1 - cl
            var_usd   = -np.percentile(pnl, alpha * 100)
            var_pct   = -np.percentile(pnl_pct, alpha * 100)
            tail_mask = pnl <= -var_usd
            cvar_usd  = -pnl[tail_mask].mean()     if tail_mask.any() else var_usd
            cvar_pct  = -pnl_pct[tail_mask].mean() if tail_mask.any() else var_pct
            rows.append({
                "Confidence level":    f"{int(cl*100)}%",
                "VaR (USD)":           f"${var_usd:.2f}",
                "VaR (% of S0)":       f"{var_pct:.1f}%",
                "CVaR / ES (USD)":     f"${cvar_usd:.2f}",
                "CVaR / ES (% of S0)": f"{cvar_pct:.1f}%",
            })
        return pd.DataFrame(rows)

    def show_risk_metrics(paths, S0, days, ticker):
        st.subheader("⚠️ Risk Metrics — VaR & CVaR")
        st.markdown(
            f"Tail-risk estimates for **{ticker}** over **{days} trading days**, "
            "derived from the full simulated final-price distribution."
        )
        risk_df = compute_risk_metrics(paths, S0)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        final   = paths[:, -1]
        pnl_pct = (final / S0 - 1) * 100
        with col1:
            var95  = -np.percentile((final - S0), 5)
            cvar95 = -(final - S0)[final - S0 <= -var95].mean()
            st.metric("VaR 95%",       f"${var95:.2f}",  f"{-var95/S0*100:.1f}% of S0")
            st.metric("CVaR 95% (ES)", f"${cvar95:.2f}", f"{-cvar95/S0*100:.1f}% of S0")
        with col2:
            prob_loss_10 = np.mean(pnl_pct < -10) * 100
            prob_loss_25 = np.mean(pnl_pct < -25) * 100
            st.metric("Prob. of >10% loss", f"{prob_loss_10:.1f}%")
            st.metric("Prob. of >25% loss", f"{prob_loss_25:.1f}%")

        with st.expander("What do these numbers mean?"):
            st.markdown("""
**VaR (Value at Risk)** answers: *"What is the maximum loss I should expect NOT to exceed, X% of the time?"*
- **VaR 95%** = there is a 5% chance of losing *more* than this amount over the forecast horizon.
- **VaR 99%** = there is a 1% chance of losing more than this amount.

**CVaR (Conditional VaR) / Expected Shortfall** answers: *"If I end up in the worst tail, what is my average loss?"*
- CVaR is always ≥ VaR. It tells you the *severity* of bad outcomes, not just a threshold.

> ⚠️ These figures are model-derived estimates. They are not guarantees and should not be the sole basis for investment decisions.
            """)

    # ── Price Target Probability ──────────────────────────────────────────────

    def show_price_target(paths, S0, days, ticker):
        st.subheader("🎯 Price Target Probability")
        st.markdown(
            "Enter a target price to see the probability and expected timing "
            "of the stock reaching it within the simulation horizon."
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            target = st.number_input(
                "Target price (USD)",
                min_value=0.01,
                value=round(S0 * 1.25, 2),
                step=0.50,
                format="%.2f",
                key="target_price_input",
            )
            direction = st.radio(
                "Direction",
                ["At or above (bullish)", "At or below (bearish)"],
                key="target_direction",
            )

        if target <= 0:
            st.warning("Enter a positive target price.")
            return

        final      = paths[:, -1]
        n_sims     = paths.shape[0]
        is_bullish = "above" in direction

        # Probability of reaching target at any point during the horizon
        if is_bullish:
            reached_mask = np.any(paths >= target, axis=1)
        else:
            reached_mask = np.any(paths <= target, axis=1)

        prob_reach  = np.mean(reached_mask) * 100
        prob_end    = (np.mean(final >= target) if is_bullish else np.mean(final <= target)) * 100
        pct_vs_s0   = (target / S0 - 1) * 100

        # Expected first-hit day for paths that do reach the target
        first_hits = []
        for path in paths:
            if is_bullish:
                hits = np.where(path >= target)[0]
            else:
                hits = np.where(path <= target)[0]
            if len(hits) > 0:
                first_hits.append(hits[0])

        avg_hit_day  = int(np.mean(first_hits))  if first_hits else None
        med_hit_day  = int(np.median(first_hits)) if first_hits else None

        with col_b:
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Prob. of hitting target",
                f"{prob_reach:.1f}%",
                help="Fraction of simulations where price touches the target at any point.",
            )
            m2.metric(
                "Prob. at end of horizon",
                f"{prob_end:.1f}%",
                help="Fraction of simulations where final price is beyond the target.",
            )
            m3.metric(
                "Target vs current",
                f"{pct_vs_s0:+.1f}%",
                f"${target:.2f} vs ${S0:.2f}",
            )

            if avg_hit_day is not None:
                st.caption(
                    f"📅 Among simulations that reach the target: "
                    f"median arrival **day {med_hit_day}**, average **day {avg_hit_day}** "
                    f"(out of {days} total trading days)."
                )
            else:
                st.caption("No simulations reached this target.")

        # Build probability curve: prob of reaching target by day X
        reach_by_day = []
        for d in range(paths.shape[1]):
            if is_bullish:
                frac = np.mean(np.any(paths[:, :d+1] >= target, axis=1))
            else:
                frac = np.mean(np.any(paths[:, :d+1] <= target, axis=1))
            reach_by_day.append(frac * 100)

        days_x = list(range(paths.shape[1]))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=days_x, y=reach_by_day,
            mode="lines",
            line=dict(color=THEME["green"] if is_bullish else THEME["red"], width=2.5),
            fill="toself",
            fillcolor=f"rgba(99,153,34,0.15)" if is_bullish else "rgba(216,90,48,0.15)",
            name=f"Prob. of {'reaching' if is_bullish else 'falling to'} ${target:.2f}",
            hovertemplate="Day %{x}<br>Probability: %{y:.1f}%<extra></extra>",
        ))

        # Mark S0 horizontal reference
        fig.add_hline(
            y=prob_reach,
            line=dict(color=THEME["amber"], width=1, dash="dot"),
            annotation_text=f"Final: {prob_reach:.1f}%",
            annotation_font_color=THEME["amber"],
        )

        fig.update_layout(
            **base_layout(
                title=f"Cumulative probability of {'reaching' if is_bullish else 'falling to'} "
                      f"${target:.2f} by trading day X  |  {ticker}",
                xaxis=dict(title="Trading days forward"),
                yaxis=dict(title="Probability (%)", ticksuffix="%", range=[0, 102]),
                height=380,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price probability distribution overlay
        st.markdown("**Final price distribution with target marker**")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=final, nbinsx=60,
            marker=dict(color=THEME["blue"], opacity=0.65,
                        line=dict(color=THEME["panel"], width=0.3)),
            name="Simulated final prices",
            hovertemplate="Price: $%{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig2.add_vline(
            x=S0,
            line=dict(color=THEME["amber"], width=1.5, dash="dot"),
            annotation_text=f"Current ${S0:.2f}",
            annotation_font_color=THEME["amber"],
        )
        fig2.add_vline(
            x=target,
            line=dict(color=THEME["green"] if is_bullish else THEME["red"], width=2.5),
            annotation_text=f"Target ${target:.2f}",
            annotation_font_color=THEME["green"] if is_bullish else THEME["red"],
        )
        fig2.update_layout(
            **base_layout(
                title=f"Final price distribution vs target — day {days}",
                xaxis=dict(title="Final price (USD)", tickprefix="$"),
                yaxis=dict(title="Number of simulations"),
                height=320,
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── AI Summary ───────────────────────────────────────────────────────────

    def build_ai_prompt(ticker, info, params, paths, pcts, days,
                        n_sims, preset_name, ewma_lambda, t_dof, start, end):
        final        = paths[:, -1]
        S0           = params["last_price"]
        p5           = float(np.percentile(final,  5))
        p25          = float(np.percentile(final, 25))
        p50          = float(np.percentile(final, 50))
        p75          = float(np.percentile(final, 75))
        p95          = float(np.percentile(final, 95))
        prob_gain    = float(np.mean(final > S0) * 100)
        worst        = float(np.min(final))
        best         = float(np.max(final))
        pnl_pct      = (final / S0 - 1) * 100
        prob_loss10  = float(np.mean(pnl_pct < -10) * 100)
        prob_loss25  = float(np.mean(pnl_pct < -25) * 100)
        var95        = float(-np.percentile(final - S0, 5))
        tail_mask    = (final - S0) <= -var95
        cvar95       = float(-(final - S0)[tail_mask].mean()) if tail_mask.any() else var95

        company_name = info.get("longName", ticker.upper())
        sector       = info.get("sector",   "Unknown sector")
        industry     = info.get("industry", "Unknown industry")

        ann_vol = params["ann_vol_ewma"]
        if ann_vol < 0.20:
            vol_regime = "low (below 20% annualised)"
        elif ann_vol < 0.40:
            vol_regime = "moderate (20–40% annualised)"
        elif ann_vol < 0.70:
            vol_regime = "high (40–70% annualised)"
        else:
            vol_regime = "very high (above 70% annualised)"

        drift = params["ann_drift_recent"]
        if drift > 0.15:
            drift_desc = f"strongly positive ({drift:+.1%} annualised)"
        elif drift > 0.02:
            drift_desc = f"mildly positive ({drift:+.1%} annualised)"
        elif drift > -0.02:
            drift_desc = f"roughly flat ({drift:+.1%} annualised)"
        elif drift > -0.15:
            drift_desc = f"mildly negative ({drift:+.1%} annualised)"
        else:
            drift_desc = f"strongly negative ({drift:+.1%} annualised)"

        spread_pct = (p95 - p5) / S0 * 100
        if spread_pct < 30:
            spread_desc = "tight"
        elif spread_pct < 80:
            spread_desc = "moderate"
        else:
            spread_desc = "very wide"

        trading_years = days / 252

        prompt = f"""You are a financial analyst assistant helping everyday retail investors understand the results of a Monte Carlo stock price simulation. Your job is to explain what the numbers mean in plain English — no jargon, no math symbols.

Here are the simulation results for {company_name} ({ticker.upper()}):

COMPANY INFO:
- Sector: {sector} / Industry: {industry}

SIMULATION SETTINGS:
- Historical data: {start} to {end} ({params['num_hist_days']} trading days)
- Forecast horizon: {days} trading days (~{trading_years:.1f} year{"s" if trading_years != 1 else ""})
- Simulations: {n_sims:,} | Preset: {preset_name} | λ: {ewma_lambda} | t-dof: {t_dof}

PRICE FORECAST:
- Current price: ${S0:.2f}
- Median forecast: ${p50:.2f} ({(p50/S0-1)*100:+.1f}%)
- 95th percentile: ${p95:.2f} ({(p95/S0-1)*100:+.1f}%)
- 5th percentile: ${p5:.2f} ({(p5/S0-1)*100:+.1f}%)
- 25th–75th range: ${p25:.2f} to ${p75:.2f}
- Best / Worst: ${best:.2f} / ${worst:.2f}
- Probability of gain: {prob_gain:.1f}%

RISK:
- VaR 95%: ${var95:.2f} ({var95/S0*100:.1f}% of investment)
- CVaR 95%: ${cvar95:.2f} ({cvar95/S0*100:.1f}% of investment)
- Prob >10% loss: {prob_loss10:.1f}% | Prob >25% loss: {prob_loss25:.1f}%

MODEL:
- Volatility regime: {vol_regime}
- Recent drift: {drift_desc}
- Outcome spread: {spread_desc} ({spread_pct:.0f}% of starting price)

Write a clear, friendly, honest summary in four bold-headed sections:
1. **What the simulation is telling you** (2–3 sentences)
2. **The range of outcomes** (explain percentiles in plain English)
3. **The risk picture** (no jargon — describe what the numbers mean practically)
4. **Important things to keep in mind** (limitations + one-sentence disclaimer)

Under 420 words. Flowing paragraphs, no bullet points. Reference numbers once then explain in human terms."""
        return prompt

    def show_ai_summary(ticker, info, params, paths, pcts, days,
                        n_sims, preset_name, ewma_lambda, t_dof, start, end):
        st.subheader("🤖 AI Summary — Plain English Breakdown")
        st.markdown("An AI-generated explanation of your simulation results, written for everyday investors.")

        groq_key = st.secrets.get("GROQ_API_KEY", "")
        if not groq_key:
            st.warning("Groq API key not configured. Add `GROQ_API_KEY` to your Streamlit secrets.")
            return

        prompt = build_ai_prompt(ticker, info, params, paths, pcts, days,
                                 n_sims, preset_name, ewma_lambda, t_dof, start, end)
        try:
            client = Groq(api_key=groq_key)
            with st.spinner("Generating AI summary…"):
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": (
                            "You are a clear, honest, and friendly financial educator. "
                            "You explain quantitative results in plain English for retail investors. "
                            "You never make price predictions or give investment advice."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=650,
                    stream=True,
                )
                summary_box = st.empty()
                full_text   = ""
                for chunk in stream:
                    delta      = chunk.choices[0].delta.content or ""
                    full_text += delta
                    summary_box.markdown(
                        f"""<div style="background:#1a1a2e;border-left:4px solid #378ADD;
                        border-radius:6px;padding:1.2rem 1.5rem;color:#e0e0e0;
                        font-size:0.97rem;line-height:1.75;white-space:pre-wrap;">{full_text}▌</div>""",
                        unsafe_allow_html=True,
                    )
                summary_box.markdown(
                    f"""<div style="background:#1a1a2e;border-left:4px solid #378ADD;
                    border-radius:6px;padding:1.2rem 1.5rem;color:#e0e0e0;
                    font-size:0.97rem;line-height:1.75;white-space:pre-wrap;">{full_text}</div>""",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"AI summary unavailable: {e}")

    # ── Downloads ─────────────────────────────────────────────────────────────

    def build_csv(paths, pcts, S0, ticker, days, start, end, params,
                  ewma_lambda, t_dof, preset_name) -> bytes:
        final   = paths[:, -1]
        pnl_pct = (final / S0 - 1) * 100
        p5, p25, p50, p75, p95 = (np.percentile(final, p) for p in [5, 25, 50, 75, 95])
        prob_profit = np.mean(final > S0) * 100

        summary_rows = [
            ["=== SIMULATION SUMMARY ===", ""],
            ["Ticker", ticker], ["Historical start", str(start)],
            ["Historical end", str(end)],
            ["Trading days in range", params["num_hist_days"]],
            ["Days simulated forward", days],
            ["Number of simulations", paths.shape[0]],
            ["Preset used", preset_name],
            ["EWMA lambda", ewma_lambda], ["Student-t dof", t_dof],
            ["Drift window (days)", params["drift_window_used"]], ["", ""],
            ["Starting price (S0)", f"${S0:.4f}"],
            ["Median forecast", f"${p50:.4f}"],
            ["Median vs S0", f"{(p50/S0-1)*100:+.2f}%"],
            ["Probability of gain", f"{prob_profit:.1f}%"],
            ["EWMA vol (ann.)", f"{params['ann_vol_ewma']*100:.2f}%"],
            ["Simple hist vol (ann.)", f"{params['ann_vol_simple']*100:.2f}%"],
            ["Recent drift (ann.)", f"{params['ann_drift_recent']*100:+.2f}%"],
            ["Full drift (ann.)", f"{params['ann_drift_full']*100:+.2f}%"], ["", ""],
            ["=== PERCENTILE TABLE ===", ""],
            ["Percentile", "Price (USD)", "vs S0 (%)"],
            ["5th",           f"${p5:.4f}",            f"{(p5/S0-1)*100:+.2f}%"],
            ["25th",          f"${p25:.4f}",           f"{(p25/S0-1)*100:+.2f}%"],
            ["50th (median)", f"${p50:.4f}",           f"{(p50/S0-1)*100:+.2f}%"],
            ["75th",          f"${p75:.4f}",           f"{(p75/S0-1)*100:+.2f}%"],
            ["95th",          f"${p95:.4f}",           f"{(p95/S0-1)*100:+.2f}%"],
            ["Worst",         f"${np.min(final):.4f}", f"{(np.min(final)/S0-1)*100:+.2f}%"],
            ["Best",          f"${np.max(final):.4f}", f"{(np.max(final)/S0-1)*100:+.2f}%"],
            ["", ""], ["=== RISK METRICS ===", ""],
            ["Confidence", "VaR (USD)", "VaR (%)", "CVaR (USD)", "CVaR (%)"],
        ]
        for cl in [0.90, 0.95, 0.99]:
            alpha     = 1 - cl
            var_usd   = -np.percentile(final - S0, alpha * 100)
            var_pct   = -np.percentile(pnl_pct, alpha * 100)
            tail_mask = (final - S0) <= -var_usd
            cvar_usd  = -(final - S0)[tail_mask].mean() if tail_mask.any() else var_usd
            cvar_pct  = -pnl_pct[tail_mask].mean()      if tail_mask.any() else var_pct
            summary_rows.append([
                f"{int(cl*100)}%",
                f"${var_usd:.4f}", f"{var_pct:.2f}%",
                f"${cvar_usd:.4f}", f"{cvar_pct:.2f}%",
            ])
        buf    = io.StringIO()
        writer = csv.writer(buf)
        for row in summary_rows:
            writer.writerow(row)
        return buf.getvalue().encode()

    def show_downloads(figs, paths, pcts, S0, ticker, days,
                       start, end, params, ewma_lambda, t_dof, preset_name):
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            html_bytes = figs["fan"].to_html(
                include_plotlyjs="cdn", full_html=True,
                config={"scrollZoom": True, "displayModeBar": True},
            ).encode("utf-8")
            st.download_button(
                label="⬇️ Download simulation chart (HTML)",
                data=html_bytes,
                file_name=f"{ticker}_montecarlo_{date.today()}.html",
                mime="text/html",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="⬇️ Download results (CSV)",
                data=build_csv(paths, pcts, S0, ticker, days, start, end,
                               params, ewma_lambda, t_dof, preset_name),
                file_name=f"{ticker}_montecarlo_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Simulation summary ────────────────────────────────────────────────────

    def show_summary(ticker, params, paths, days, start, end, ewma_lambda, t_dof):
        final = paths[:, -1]
        S0    = params["last_price"]
        p5, p25, p50, p75, p95 = (np.percentile(final, p) for p in [5, 25, 50, 75, 95])
        prob_profit = np.mean(final > S0) * 100
        worst, best = np.min(final), np.max(final)

        st.subheader("📊 Simulation Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Starting price (S0)",    f"${S0:.2f}")
        col2.metric("Median forecast",        f"${p50:.2f}", f"{(p50/S0-1):+.1%}")
        col3.metric("Probability of gain",    f"{prob_profit:.1f}%")
        col4.metric("EWMA volatility (ann.)", f"{params['ann_vol_ewma']:.1%}")

        with st.expander("Full percentile table"):
            summary_df = pd.DataFrame({
                "Percentile":  ["5th", "25th", "50th (median)", "75th", "95th", "Worst path", "Best path"],
                "Price (USD)": [p5, p25, p50, p75, p95, worst, best],
                "vs S0":       [(v/S0-1) for v in [p5, p25, p50, p75, p95, worst, best]],
            })
            summary_df["Price (USD)"] = summary_df["Price (USD)"].map("${:.2f}".format)
            summary_df["vs S0"]       = summary_df["vs S0"].map("{:+.1%}".format)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with st.expander("Model parameters"):
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Historical data range | {start} → {end} |
| Trading days in range | {params['num_hist_days']} |
| Full-window drift (ann.) | {params['ann_drift_full']:+.1%} |
| Recent-window drift (ann., last {params['drift_window_used']}d) | {params['ann_drift_recent']:+.1%} |
| Simple historical vol (ann.) | {params['ann_vol_simple']:.1%} |
| EWMA vol (ann., λ={ewma_lambda}) | {params['ann_vol_ewma']:.1%} |
| Simulated vol (ann., scaled) | {params['ann_vol_scaled']:.1%} |
| Student-t degrees of freedom | {t_dof} |
| Days simulated forward | {days} |
            """)

    # ── Charts ────────────────────────────────────────────────────────────────

    def build_fan_chart(ticker, closes, params, paths, pcts, days, n_sims, info, start, end):
        S0     = params["last_price"]
        days_x = [int(i) for i in range(days + 1)]
        name   = info.get("longName", ticker.upper())
        fig    = go.Figure()

        sample_n = min(n_sims, 80)
        idx      = np.random.choice(n_sims, sample_n, replace=False)
        for i in idx:
            fig.add_trace(go.Scatter(
                x=days_x, y=paths[i], mode="lines",
                line=dict(color="rgba(255,255,255,0.04)", width=0.8),
                hoverinfo="skip", showlegend=False,
            ))

        days_fwd = list(days_x)
        days_rev = list(days_x[::-1])
        fig.add_trace(go.Scatter(
            x=days_fwd + days_rev,
            y=list(pcts["p95"]) + list(pcts["p5"][::-1]),
            fill="toself", fillcolor="rgba(55,138,221,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="5–95th %ile band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=days_fwd + days_rev,
            y=list(pcts["p75"]) + list(pcts["p25"][::-1]),
            fill="toself", fillcolor="rgba(55,138,221,0.22)",
            line=dict(color="rgba(0,0,0,0)"), name="25–75th %ile band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=days_x, y=pcts["p95"], mode="lines",
            line=dict(color=THEME["green"], width=1.5, dash="dash"),
            name="95th percentile",
            hovertemplate="Day %{x}<br>95th: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=days_x, y=pcts["p50"], mode="lines",
            line=dict(color=THEME["blue"], width=2.5), name="Median",
            hovertemplate="Day %{x}<br>Median: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=days_x, y=pcts["p5"], mode="lines",
            line=dict(color=THEME["red"], width=1.5, dash="dash"),
            name="5th percentile",
            hovertemplate="Day %{x}<br>5th: $%{y:.2f}<extra></extra>",
        ))
        fig.add_hline(
            y=S0, line=dict(color=THEME["amber"], width=1.2, dash="dot"),
            annotation_text=f"S0  ${S0:.2f}", annotation_font_color=THEME["amber"],
        )
        fig.update_layout(
            **base_layout(
                title=f"{name} ({ticker}) — {n_sims:,} simulations · {days} trading days forward  |  "
                      f"EWMA vol {params['ann_vol_ewma']:.1%}  ·  Recent drift {params['ann_drift_recent']:+.1%}",
                xaxis=dict(title="Trading days forward"),
                yaxis=dict(title="Simulated price (USD)", tickprefix="$"),
                height=480,
            ),
            dragmode="zoom",
        )
        return fig

    def build_forecast_chart(ticker, closes, params, paths, pcts, days, info, start, end, actual_closes=None):
        S0             = params["last_price"]
        name           = info.get("longName", ticker.upper())
        hist_dates     = [d.date() if hasattr(d, "date") else d for d in closes.index]
        forecast_dates = add_trading_days(hist_dates[-1], days)
        median_path    = find_median_path(paths, pcts)
        bridge_dates   = [hist_dates[-1]] + forecast_dates
        conf_low       = np.concatenate([[S0], pcts["p5"][1:]])
        conf_high      = np.concatenate([[S0], pcts["p95"][1:]])

        fig = go.Figure()
        band_x = [str(d) for d in bridge_dates] + [str(d) for d in bridge_dates[::-1]]
        fig.add_trace(go.Scatter(
            x=band_x, y=list(conf_high) + list(conf_low[::-1]),
            fill="toself", fillcolor="rgba(76,175,120,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="5–95th %ile band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[str(d) for d in hist_dates], y=closes.values, mode="lines",
            line=dict(color=THEME["hist_col"], width=2), name="History",
            hovertemplate="%{x}<br>Close: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[str(d) for d in bridge_dates], y=median_path, mode="lines",
            line=dict(color=THEME["fore_col"], width=2),
            name=f"Most-likely forecast ({days}d)",
            hovertemplate="%{x}<br>Forecast: $%{y:.2f}<extra></extra>",
        ))
        if actual_closes is not None and len(actual_closes) > 1:
            actual_dates = [d.date() if hasattr(d, "date") else d for d in actual_closes.index]
            fig.add_trace(go.Scatter(
                x=[str(d) for d in actual_dates], y=actual_closes.values, mode="lines",
                line=dict(color="#E03C3C", width=2), name="Actual price (post-simulation)",
                hovertemplate="%{x}<br>Actual: $%{y:.2f}<extra></extra>",
            ))
        fig.add_shape(
            type="line", x0=str(hist_dates[-1]), x1=str(hist_dates[-1]),
            y0=0, y1=1, xref="x", yref="paper",
            line=dict(color=THEME["muted"], width=1.2, dash="dash"),
        )
        fig.add_annotation(
            x=str(hist_dates[-1]), y=1, xref="x", yref="paper",
            text="Forecast start", showarrow=False,
            font=dict(color=THEME["muted"], size=10), yanchor="bottom",
        )
        fig.update_layout(
            **base_layout(
                title=f"{name} ({ticker}) — History & Most-Likely Forecast  |  "
                      f"{hist_dates[0]} → {forecast_dates[-1]}",
                xaxis=dict(title="Date", type="date"),
                yaxis=dict(title="Price (USD)", tickprefix="$"),
                height=480,
            ),
            dragmode="zoom",
        )
        fig.update_layout(hovermode="x")
        return fig

    def build_histogram_chart(ticker, paths, pcts, S0, days):
        final = paths[:, -1]
        p5    = np.percentile(final,  5)
        p50   = np.percentile(final, 50)
        p95   = np.percentile(final, 95)
        fig   = go.Figure()
        fig.add_trace(go.Histogram(
            x=final, nbinsx=60,
            marker=dict(color=THEME["blue"], opacity=0.75,
                        line=dict(color=THEME["panel"], width=0.3)),
            name="Simulated final prices",
            hovertemplate="Price: $%{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        for val, label, color in [
            (p5,  "5th %ile",  THEME["red"]),
            (p50, "Median",    THEME["blue"]),
            (p95, "95th %ile", THEME["green"]),
            (S0,  "S0",        THEME["amber"]),
        ]:
            fig.add_vline(
                x=val, line=dict(color=color, width=1.8,
                                 dash="dash" if val != p50 else "solid"),
                annotation_text=f"{label} ${val:.2f}",
                annotation_font_color=color, annotation_font_size=10,
            )
        fig.update_layout(**base_layout(
            title=f"Final price distribution — day {days}",
            xaxis=dict(title="Price (USD)", tickprefix="$"),
            yaxis=dict(title="Number of simulations"),
            height=400,
        ))
        return fig

    def build_returns_chart(ticker, params, start, end):
        rets   = params["log_returns"] * 100
        mean_r = float(rets.mean())
        std_r  = float(rets.std())
        fig    = go.Figure()
        fig.add_trace(go.Histogram(
            x=rets, nbinsx=60,
            marker=dict(color=THEME["amber"], opacity=0.75,
                        line=dict(color=THEME["panel"], width=0.3)),
            name="Daily log-returns",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        for val, label, color in [
            (0,              "Zero",                    THEME["muted"]),
            (mean_r,         f"Mean {mean_r:+.2f}%",   THEME["green"]),
            (mean_r - std_r, f"−1σ {mean_r-std_r:.2f}%", THEME["red"]),
            (mean_r + std_r, f"+1σ {mean_r+std_r:.2f}%", THEME["red"]),
        ]:
            fig.add_vline(
                x=val, line=dict(color=color, width=1.4, dash="dash"),
                annotation_text=label,
                annotation_font_color=color, annotation_font_size=10,
            )
        fig.update_layout(**base_layout(
            title=f"Historical daily log-returns  |  {start} → {end}",
            xaxis=dict(title="Daily return (%)", ticksuffix="%"),
            yaxis=dict(title="Frequency"),
            height=400,
        ))
        return fig

    # ── Main app body ─────────────────────────────────────────────────────────

    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        elif (end_date - start_date).days < 30:
            st.error("Date range must span at least 30 calendar days.")
        else:
            try:
                closes, info = fetch_data(ticker, start_date, end_date)
                params = compute_parameters(
                    closes, drift_override=drift_override,
                    vol_scale=vol_scale, ewma_lambda=ewma_lambda,
                    drift_window=drift_window,
                )
                with st.spinner("Running simulations…"):
                    paths = run_simulation(params, days_forward, num_simulations, t_dof)

                pcts = compute_percentiles(paths)
                S0   = params["last_price"]

                show_summary(ticker, params, paths, days_forward,
                             start_date, end_date, ewma_lambda, t_dof)
                st.divider()
                show_risk_metrics(paths, S0, days_forward, ticker)
                st.divider()
                show_price_target(paths, S0, days_forward, ticker)
                st.divider()
                show_ai_summary(
                    ticker, info, params, paths, pcts, days_forward,
                    num_simulations, preset_name, ewma_lambda, t_dof,
                    start_date, end_date,
                )
                st.divider()
                st.subheader("📉 Charts")

                fig_fan = build_fan_chart(
                    ticker, closes, params, paths, pcts,
                    days_forward, num_simulations, info, start_date, end_date,
                )
                st.plotly_chart(fig_fan, use_container_width=True)

                actual_closes = fetch_actual_data(ticker, end_date)
                fig_forecast  = build_forecast_chart(
                    ticker, closes, params, paths, pcts, days_forward,
                    info, start_date, end_date, actual_closes=actual_closes,
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(
                        build_histogram_chart(ticker, paths, pcts, S0, days_forward),
                        use_container_width=True,
                    )
                with col_r:
                    st.plotly_chart(
                        build_returns_chart(ticker, params, start_date, end_date),
                        use_container_width=True,
                    )

                st.divider()
                show_downloads(
                    {"fan": fig_fan}, paths, pcts, S0, ticker, days_forward,
                    start_date, end_date, params, ewma_lambda, t_dof, preset_name,
                )

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    else:
        st.info("👈 Configure your parameters in the sidebar, then click **▶ Run Simulation**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO MODE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_port:

    st.title("💼 Portfolio Monte Carlo")
    st.markdown(
        "Simulate the future value of a **multi-asset portfolio**. "
        "Enter up to 8 tickers with allocation weights — the model runs independent "
        "GBM simulations for each asset and combines them into a portfolio value path."
    )

    # ── Portfolio inputs ──────────────────────────────────────────────────────

    st.subheader("Portfolio Setup")

    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    with pcol1:
        port_value = st.number_input(
            "Total portfolio value (USD)",
            min_value=100.0, max_value=100_000_000.0,
            value=10_000.0, step=1000.0, format="%.2f",
        )
    with pcol2:
        port_days = st.slider(
            "Forecast horizon (trading days)",
            min_value=21, max_value=756, value=252, step=21,
        )
    with pcol3:
        port_sims = st.slider(
            "Simulations",
            min_value=100, max_value=10000, value=500, step=100,
        )

    pcol4, pcol5 = st.columns(2)
    with pcol4:
        port_start = st.date_input(
            "Historical start date",
            value=date.today() - timedelta(days=365),
            max_value=date.today() - timedelta(days=31),
            key="port_start",
        )
    with pcol5:
        port_end = st.date_input(
            "Historical end date",
            value=date.today(),
            min_value=port_start + timedelta(days=30),
            max_value=date.today(),
            key="port_end",
        )

    st.markdown("**Assets & Weights**")
    st.caption("Weights must sum to 100%. Adjust to match your allocation.")

    # Dynamic asset rows
    n_assets = st.number_input("Number of assets", min_value=2, max_value=8, value=3, step=1)

    asset_rows = []
    default_tickers  = ["AAPL", "NVDA", "SPY", "MSFT", "TSLA", "BND", "GLD", "AMZN"]
    default_weights  = [40, 30, 30, 0, 0, 0, 0, 0]

    header_cols = st.columns([2, 1, 1])
    header_cols[0].markdown("**Ticker**")
    header_cols[1].markdown("**Weight (%)**")
    header_cols[2].markdown("**Preset**")

    for i in range(int(n_assets)):
        c1, c2, c3 = st.columns([2, 1, 1])
        t = c1.text_input(
            f"Ticker {i+1}", value=default_tickers[i],
            key=f"port_ticker_{i}", label_visibility="collapsed",
        ).strip().upper()
        w = c2.number_input(
            f"Weight {i+1}", min_value=0.0, max_value=100.0,
            value=float(default_weights[i]), step=5.0, format="%.1f",
            key=f"port_weight_{i}", label_visibility="collapsed",
        )
        p = c3.selectbox(
            f"Preset {i+1}", options=list(PRESETS.keys()), index=2,
            key=f"port_preset_{i}", label_visibility="collapsed",
        )
        asset_rows.append({"ticker": t, "weight": w, "preset": p})

    total_weight = sum(r["weight"] for r in asset_rows)
    if abs(total_weight - 100.0) > 0.01:
        st.warning(f"Weights sum to **{total_weight:.1f}%** — must equal 100%.")
    else:
        st.success("✅ Weights sum to 100%")

    port_run_btn = st.button("▶ Run Portfolio Simulation", type="primary",
                             use_container_width=False)

    # ── Portfolio engine ──────────────────────────────────────────────────────

    if port_run_btn:
        if abs(total_weight - 100.0) > 0.01:
            st.error("Please adjust weights to sum to exactly 100% before running.")
        elif port_start >= port_end:
            st.error("Start date must be before end date.")
        else:
            try:
                asset_data   = []
                failed       = []

                with st.spinner("Fetching data for all assets…"):
                    for row in asset_rows:
                        if row["weight"] == 0:
                            continue
                        try:
                            closes, info = fetch_data(row["ticker"], port_start, port_end)
                            preset       = PRESETS[row["preset"]]
                            params       = compute_parameters(
                                closes,
                                vol_scale    = preset["vol_scale"]    if preset else 1.0,
                                ewma_lambda  = preset["ewma_lambda"]  if preset else 0.94,
                                drift_window = preset["drift_window"] if preset else 63,
                            )
                            asset_data.append({
                                "ticker":  row["ticker"],
                                "weight":  row["weight"] / 100.0,
                                "closes":  closes,
                                "info":    info,
                                "params":  params,
                                "preset":  row["preset"],
                            })
                        except Exception as err:
                            failed.append(f"{row['ticker']}: {err}")

                if failed:
                    for f in failed:
                        st.error(f)

                if not asset_data:
                    st.error("No valid assets. Check tickers and date range.")
                else:
                    with st.spinner("Running portfolio simulations…"):
                        # Simulate each asset independently; combine by weight × allocation
                        portfolio_paths = np.zeros((port_sims, port_days + 1))
                        asset_paths_all = {}

                        for ad in asset_data:
                            t_dof_asset = (
                                PRESETS[ad["preset"]]["t_dof"]
                                if PRESETS[ad["preset"]] else 5
                            )
                            paths_asset = run_simulation(ad["params"], port_days, port_sims, t_dof_asset)
                            # Normalise to return-relative (paths / S0 gives return multiplier)
                            S0_asset      = ad["params"]["last_price"]
                            alloc_value   = port_value * ad["weight"]
                            port_paths_i  = (paths_asset / S0_asset) * alloc_value
                            portfolio_paths += port_paths_i
                            asset_paths_all[ad["ticker"]] = {
                                "paths":   paths_asset,
                                "params":  ad["params"],
                                "info":    ad["info"],
                                "closes":  ad["closes"],
                                "weight":  ad["weight"],
                                "alloc":   alloc_value,
                                "color":   PORTFOLIO_COLORS[len(asset_paths_all) % len(PORTFOLIO_COLORS)],
                            }

                    pcts_port = compute_percentiles(portfolio_paths)
                    final_port = portfolio_paths[:, -1]
                    p5_p, p50_p, p95_p = (np.percentile(final_port, p) for p in [5, 50, 95])
                    prob_gain_p = np.mean(final_port > port_value) * 100

                    # ── Portfolio summary metrics ─────────────────────────────
                    st.subheader("📊 Portfolio Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Starting value",    f"${port_value:,.2f}")
                    m2.metric("Median forecast",   f"${p50_p:,.2f}",
                              f"{(p50_p/port_value-1):+.1%}")
                    m3.metric("Prob. of gain",     f"{prob_gain_p:.1f}%")
                    m4.metric("95th percentile",   f"${p95_p:,.2f}",
                              f"{(p95_p/port_value-1):+.1%}")

                    # Asset breakdown table
                    with st.expander("Asset breakdown"):
                        breakdown_rows = []
                        for tk, ad in asset_paths_all.items():
                            final_a = ad["paths"][:, -1]
                            S0_a    = ad["params"]["last_price"]
                            med_a   = np.percentile(final_a, 50)
                            breakdown_rows.append({
                                "Ticker":        tk,
                                "Weight":        f"{ad['weight']*100:.1f}%",
                                "Allocation":    f"${ad['alloc']:,.2f}",
                                "Current price": f"${S0_a:.2f}",
                                "Median forecast": f"${med_a:.2f}",
                                "Median return": f"{(med_a/S0_a-1)*100:+.1f}%",
                                "EWMA vol (ann.)": f"{ad['params']['ann_vol_ewma']*100:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(breakdown_rows),
                                     use_container_width=True, hide_index=True)

                    st.divider()

                    # ── Portfolio fan chart ───────────────────────────────────
                    st.subheader("📉 Portfolio Charts")
                    days_x   = [int(i) for i in range(port_days + 1)]
                    days_fwd = list(days_x)
                    days_rev = list(days_x[::-1])

                    fig_port = go.Figure()

                    # Ghost paths
                    sample_n = min(port_sims, 60)
                    idx      = np.random.choice(port_sims, sample_n, replace=False)
                    for i in idx:
                        fig_port.add_trace(go.Scatter(
                            x=days_x, y=portfolio_paths[i], mode="lines",
                            line=dict(color="rgba(255,255,255,0.04)", width=0.6),
                            hoverinfo="skip", showlegend=False,
                        ))

                    # Bands
                    fig_port.add_trace(go.Scatter(
                        x=days_fwd + days_rev,
                        y=list(pcts_port["p95"]) + list(pcts_port["p5"][::-1]),
                        fill="toself", fillcolor="rgba(55,138,221,0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="5–95th %ile band", hoverinfo="skip",
                    ))
                    fig_port.add_trace(go.Scatter(
                        x=days_fwd + days_rev,
                        y=list(pcts_port["p75"]) + list(pcts_port["p25"][::-1]),
                        fill="toself", fillcolor="rgba(55,138,221,0.22)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="25–75th %ile band", hoverinfo="skip",
                    ))

                    # Percentile lines
                    for vals, label, color, dash in [
                        (pcts_port["p95"], "95th percentile", THEME["green"], "dash"),
                        (pcts_port["p50"], "Median",          THEME["blue"],  "solid"),
                        (pcts_port["p5"],  "5th percentile",  THEME["red"],   "dash"),
                    ]:
                        fig_port.add_trace(go.Scatter(
                            x=days_x, y=vals, mode="lines",
                            line=dict(color=color, width=2.0 if dash=="solid" else 1.5, dash=dash),
                            name=label,
                            hovertemplate=f"Day %{{x}}<br>{label}: $%{{y:,.2f}}<extra></extra>",
                        ))

                    fig_port.add_hline(
                        y=port_value,
                        line=dict(color=THEME["amber"], width=1.2, dash="dot"),
                        annotation_text=f"Starting value  ${port_value:,.2f}",
                        annotation_font_color=THEME["amber"],
                    )

                    tickers_str = " + ".join(
                        f"{tk} ({ad['weight']*100:.0f}%)"
                        for tk, ad in asset_paths_all.items()
                    )
                    fig_port.update_layout(
                        **base_layout(
                            title=f"Portfolio — {tickers_str}  |  "
                                  f"{port_sims:,} simulations · {port_days} trading days",
                            xaxis=dict(title="Trading days forward"),
                            yaxis=dict(title="Portfolio value (USD)", tickprefix="$"),
                            height=500,
                        ),
                        dragmode="zoom",
                    )
                    st.plotly_chart(fig_port, use_container_width=True)

                    # ── Individual asset history + forecast + actual overlay ────
                    fig_assets = go.Figure()

                    for tk, ad in asset_paths_all.items():
                        pcts_a  = compute_percentiles(ad["paths"])
                        S0_a    = ad["params"]["last_price"]
                        color  = ad["color"]
                        closes_a= ad["closes"]

                        # Historical dates and prices (normalised to % return)
                        hist_dates_a  = [d.date() if hasattr(d, "date") else d
                                         for d in closes_a.index]
                        hist_ret_a    = (closes_a.values / closes_a.values[0] - 1) * 100

                        # Forecast dates and median path (normalised to % return from S0)
                        forecast_dates_a = add_trading_days(hist_dates_a[-1], port_days)
                        bridge_dates_a   = [hist_dates_a[-1]] + forecast_dates_a
                        median_path_a    = find_median_path(ad["paths"], pcts_a)
                        median_ret_a     = (median_path_a / S0_a - 1) * 100

                        # 5–95 band (normalised)
                        conf_low_a  = np.concatenate([[0.0], (pcts_a["p5"][1:]  / S0_a - 1) * 100])
                        conf_high_a = np.concatenate([[0.0], (pcts_a["p95"][1:] / S0_a - 1) * 100])
                        band_xa = [str(d) for d in bridge_dates_a] + [str(d) for d in bridge_dates_a[::-1]]
                        band_ya = list(conf_high_a) + list(conf_low_a[::-1])
                        fig_assets.add_trace(go.Scatter(
                            x=band_xa, y=band_ya,
                            fill="toself",
                            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False, hoverinfo="skip",
                        ))

                        # Historical price line
                        fig_assets.add_trace(go.Scatter(
                            x=[str(d) for d in hist_dates_a],
                            y=hist_ret_a,
                            mode="lines",
                            line=dict(color=color, width=2),
                            name=f"{tk} ({ad['weight']*100:.0f}%) — History",
                            hovertemplate=f"%{{x}}<br>{tk} history: %{{y:+.1f}}%<extra></extra>",
                        ))

                        # Forecast median line
                        fig_assets.add_trace(go.Scatter(
                            x=[str(d) for d in bridge_dates_a],
                            y=median_ret_a,
                            mode="lines",
                            line=dict(color=color, width=1.8, dash="dash"),
                            name=f"{tk} — Forecast",
                            hovertemplate=f"%{{x}}<br>{tk} forecast: %{{y:+.1f}}%<extra></extra>",
                            showlegend=True,
                        ))

                        # Actual post-simulation price (dashed, fetched live)
                        actual_a = fetch_actual_data(tk, port_end)
                        if actual_a is not None and len(actual_a) > 1:
                            actual_dates_a = [d.date() if hasattr(d, "date") else d
                                              for d in actual_a.index]
                            # Normalise actual vs S0 (same base as forecast)
                            actual_ret_a = (actual_a.values / S0_a - 1) * 100
                            fig_assets.add_trace(go.Scatter(
                                x=[str(d) for d in actual_dates_a],
                                y=actual_ret_a,
                                mode="lines",
                                line=dict(color=color, width=2, dash="dot"),
                                name=f"{tk} — Actual (post-simulation)",
                                hovertemplate=f"%{{x}}<br>{tk} actual: %{{y:+.1f}}%<extra></extra>",
                            ))

                    # Vertical divider at forecast start (= port_end)
                    fig_assets.add_shape(
                        type="line",
                        x0=str(port_end), x1=str(port_end),
                        y0=0, y1=1, xref="x", yref="paper",
                        line=dict(color=THEME["muted"], width=1.2, dash="dash"),
                    )
                    fig_assets.add_annotation(
                        x=str(port_end), y=1, xref="x", yref="paper",
                        text="Forecast start", showarrow=False,
                        font=dict(color=THEME["muted"], size=10), yanchor="bottom",
                    )
                    fig_assets.add_hline(
                        y=0, line=dict(color=THEME["muted"], width=1, dash="dot"),
                    )
                    fig_assets.update_layout(
                        **base_layout(
                            title="Individual asset — History · Forecast (dashed) · Actual (dotted)",
                            xaxis=dict(title="Date", type="date"),
                            yaxis=dict(title="Return vs starting price (%)", ticksuffix="%"),
                            height=480,
                        ),
                        dragmode="zoom",
                    )
                    fig_assets.update_layout(hovermode="x")
                    st.plotly_chart(fig_assets, use_container_width=True)

                    # ── Portfolio final value distribution ────────────────────
                    fig_phist = go.Figure()
                    fig_phist.add_trace(go.Histogram(
                        x=final_port, nbinsx=60,
                        marker=dict(color=THEME["blue"], opacity=0.75,
                                    line=dict(color=THEME["panel"], width=0.3)),
                        name="Portfolio final values",
                        hovertemplate="Value: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
                    ))
                    for val, label, color in [
                        (p5_p,       "5th %ile",  THEME["red"]),
                        (p50_p,      "Median",    THEME["blue"]),
                        (p95_p,      "95th %ile", THEME["green"]),
                        (port_value, "Start",     THEME["amber"]),
                    ]:
                        fig_phist.add_vline(
                            x=val,
                            line=dict(color=color, width=1.8,
                                      dash="dash" if val != p50_p else "solid"),
                            annotation_text=f"{label} ${val:,.0f}",
                            annotation_font_color=color,
                        )
                    fig_phist.update_layout(**base_layout(
                        title=f"Portfolio final value distribution — day {port_days}",
                        xaxis=dict(title="Portfolio value (USD)", tickprefix="$"),
                        yaxis=dict(title="Number of simulations"),
                        height=380,
                    ))
                    st.plotly_chart(fig_phist, use_container_width=True)

                    # ── Portfolio VaR / CVaR ──────────────────────────────────
                    st.divider()
                    st.subheader("⚠️ Portfolio Risk Metrics")
                    pnl_port = final_port - port_value
                    pnl_pct_port = (final_port / port_value - 1) * 100
                    port_risk_rows = []
                    for cl in [0.90, 0.95, 0.99]:
                        alpha     = 1 - cl
                        var_usd   = -np.percentile(pnl_port, alpha * 100)
                        var_pct   = -np.percentile(pnl_pct_port, alpha * 100)
                        tail_mask = pnl_port <= -var_usd
                        cvar_usd  = -pnl_port[tail_mask].mean()     if tail_mask.any() else var_usd
                        cvar_pct  = -pnl_pct_port[tail_mask].mean() if tail_mask.any() else var_pct
                        port_risk_rows.append({
                            "Confidence level":    f"{int(cl*100)}%",
                            "VaR (USD)":           f"${var_usd:,.2f}",
                            "VaR (% of portfolio)": f"{var_pct:.1f}%",
                            "CVaR / ES (USD)":     f"${cvar_usd:,.2f}",
                            "CVaR / ES (% of portfolio)": f"{cvar_pct:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(port_risk_rows),
                                 use_container_width=True, hide_index=True)

                    rm1, rm2, rm3, rm4 = st.columns(4)
                    var95_p  = -np.percentile(pnl_port, 5)
                    cvar95_p = -pnl_port[pnl_port <= -var95_p].mean()
                    rm1.metric("VaR 95%",           f"${var95_p:,.2f}",
                               f"{-var95_p/port_value*100:.1f}% of portfolio")
                    rm2.metric("CVaR 95%",           f"${cvar95_p:,.2f}",
                               f"{-cvar95_p/port_value*100:.1f}% of portfolio")
                    rm3.metric("Prob. of >10% loss", f"{np.mean(pnl_pct_port < -10)*100:.1f}%")
                    rm4.metric("Prob. of >25% loss", f"{np.mean(pnl_pct_port < -25)*100:.1f}%")

            except Exception as e:
                st.error(f"Unexpected error: {e}")

    else:
        st.info("👆 Set up your portfolio above and click **▶ Run Portfolio Simulation**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:

    st.title("📖 About & Methodology")

    st.markdown("""
This tool simulates future stock price paths using a **Monte Carlo method** built on
**Geometric Brownian Motion (GBM)** — the same mathematical foundation used in the
Black-Scholes options pricing model and quantitative risk management.

---

## How it works — step by step

### 1. Fetch historical data
Daily adjusted closing prices are downloaded from Yahoo Finance via `yfinance`.

### 2. Compute log returns
> **r_t = ln(P_t / P_{t-1})**

Log returns are additive across time and more closely approximate a normal distribution.

### 3. Estimate volatility — EWMA
> **σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_t**

The decay factor **λ** (default 0.94, the JP Morgan RiskMetrics standard) controls
how quickly older observations lose influence.

### 4. Estimate drift — recent window
Drift **μ** is estimated from the most recent N trading days (default 63 ≈ 3 months),
making the model sensitive to current momentum rather than multi-year history.

### 5. Simulate price paths — GBM with fat tails
> **S_{t+1} = S_t · exp( (μ − ½σ²) + σ · Z_t )**

Random shocks **Z_t** are drawn from a **Student's t-distribution** (default dof=5),
which has heavier tails than the normal — capturing the excess kurtosis observed in
real equity returns.

### 6. Portfolio mode
Each asset is simulated independently using its own calibrated parameters. Portfolio
value at each time step is the weighted sum of individual asset values, where weights
are applied to the dollar allocation (e.g. 40% of $10,000 = $4,000 allocated to AAPL).

### 7. Price target probability
For a user-specified price target, the model computes:
- The fraction of simulations where the stock **reaches** the target at any point
- The fraction where the stock **ends** beyond the target at the horizon
- A cumulative probability curve showing how likelihood builds over time
- The median and average first-hit day among paths that reach the target

---

## Risk metrics

| Metric | What it answers |
|---|---|
| **VaR** | "What is the worst loss I should expect NOT to exceed, X% of the time?" |
| **CVaR / ES** | "If I end up in the worst X% of outcomes, what is my average loss?" |

---

## Presets

| Preset | Best for | Key settings |
|---|---|---|
| **Conservative** | Large-cap, low-beta (JNJ, KO) | Low vol, slow decay, near-normal tails |
| **Balanced** | Most equities | λ=0.94, dof=5, 63-day drift |
| **Aggressive** | High-beta / speculative (TSLA, MSTR) | 1.5× vol, fast decay, very fat tails |
| **Momentum** | Stocks in strong near-term trend | 21-day drift window |

---

## Limitations & disclaimers

- GBM assumes constant volatility and drift — real markets exhibit clustering, jumps, and regime changes.
- Past price behaviour does not guarantee future results.
- The median forecast is not a prediction — half of all simulated paths end above it, half below.
- Portfolio simulations assume **zero correlation** between assets (each is simulated independently). This is a simplification: in reality, assets often move together, especially during market stress.
- This tool is for **educational and analytical purposes only** and does not constitute financial advice.

---

## Technology

**Python · Streamlit · yfinance · NumPy · SciPy · Plotly · Groq (Llama 3.3 70B)**
    """)