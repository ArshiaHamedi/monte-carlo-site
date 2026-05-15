"""
monte carlo stock price simulation - streamlit app
====================================================
all data derived from yfinance and cached for performance
run with:
    streamlit run mcarlo_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
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

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_sim, tab_about = st.tabs(["📈 Simulator", "📖 About & Methodology"])


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
            index=2,  # default: Balanced
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
            help="1.0 = model vol. 2.0 = double, 0.5 = half.",
        )
        ewma_lambda = st.slider(
            "EWMA decay factor (λ)",
            min_value=0.80, max_value=0.99, step=0.01,
            value=_val("ewma_lambda", 0.94),
            help=(
                "Controls how fast older returns decay. "
                "0.94 = RiskMetrics standard. "
                "0.99 = slower decay (closer to simple historical vol)."
            ),
        )
        t_dof = st.slider(
            "Student-t degrees of freedom",
            min_value=3, max_value=30, step=1,
            value=_val("t_dof", 5),
            help="Lower = fatter tails. 3–6 is realistic for equities. 30 ≈ normal.",
        )
        drift_window = st.slider(
            "Recent drift window (trading days)",
            min_value=21, max_value=252, step=21,
            value=_val("drift_window", 63),
            help="Number of recent trading days used to estimate drift. 63 ≈ 3 months.",
        )

        use_drift_override = st.checkbox("Override drift manually?", value=False)
        drift_override = None
        if use_drift_override:
            drift_override = st.number_input(
                "Annual drift (e.g. 0.10 = +10%)",
                min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f",
            )

        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    # ── Core functions ────────────────────────────────────────────────────────

    def apply_grid(ax, date_axis=False):
        GRID_MAJOR = "#2e2e2e"
        GRID_MINOR = "#222222"
        ax.grid(which="major", color=GRID_MAJOR, linewidth=0.7, linestyle="-",  zorder=0)
        ax.grid(which="minor", color=GRID_MINOR, linewidth=0.4, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        if date_axis:
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        else:
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    def add_trading_days(from_date: date, n_days: int) -> list:
        dates, cursor = [], from_date
        while len(dates) < n_days:
            cursor += timedelta(days=1)
            if cursor.weekday() < 5:
                dates.append(cursor)
        return dates

    @st.cache_data(show_spinner="Fetching historical data from yfinance…")
    def fetch_data(ticker: str, start: date, end: date):
        """Cached: same ticker+range won't re-hit yfinance on every widget change."""
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
                f"Only {len(closes)} trading days in range. Need at least 20 — widen the date range."
            )
        info = {}
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            pass
        return closes, info

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
- A stock with VaR 95% = $5 and CVaR 95% = $12 has a heavier tail than one where both values are close.

> ⚠️ These figures are model-derived estimates. They are not guarantees and should not be the sole basis for investment decisions.
            """)

    # ── AI Summary ───────────────────────────────────────────────────────────

    def build_ai_prompt(ticker, info, params, paths, pcts, days,
                        n_sims, preset_name, ewma_lambda, t_dof, start, end):
        """Assemble all simulation outputs into a structured prompt for Llama 3."""
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
            vol_regime = "low (below 20% annualised — typical of large-cap defensives)"
        elif ann_vol < 0.40:
            vol_regime = "moderate (20–40% annualised — typical of most equities)"
        elif ann_vol < 0.70:
            vol_regime = "high (40–70% annualised — typical of growth or speculative names)"
        else:
            vol_regime = "very high (above 70% annualised — extremely speculative)"

        drift = params["ann_drift_recent"]
        if drift > 0.15:
            drift_desc = f"strongly positive ({drift:+.1%} annualised) — the stock has had strong recent momentum"
        elif drift > 0.02:
            drift_desc = f"mildly positive ({drift:+.1%} annualised)"
        elif drift > -0.02:
            drift_desc = f"roughly flat ({drift:+.1%} annualised) — little directional trend recently"
        elif drift > -0.15:
            drift_desc = f"mildly negative ({drift:+.1%} annualised)"
        else:
            drift_desc = f"strongly negative ({drift:+.1%} annualised) — the stock has been under significant pressure"

        spread_pct = (p95 - p5) / S0 * 100
        if spread_pct < 30:
            spread_desc = "tight — the range of outcomes is relatively narrow"
        elif spread_pct < 80:
            spread_desc = "moderate — a reasonable spread of outcomes"
        else:
            spread_desc = "very wide — outcomes are highly uncertain and span a large range"

        trading_years = days / 252

        prompt = f"""You are a financial analyst assistant helping everyday retail investors understand the results of a Monte Carlo stock price simulation. Your job is to explain what the numbers mean in plain English — no jargon, no math symbols. Write as if you are a knowledgeable friend explaining this to someone who has never traded before, but is curious and intelligent.

Here are the simulation results for {company_name} ({ticker.upper()}):

COMPANY INFO:
- Sector: {sector}
- Industry: {industry}

SIMULATION SETTINGS:
- Historical data used: {start} to {end} ({params['num_hist_days']} trading days)
- Forecast horizon: {days} trading days (~{trading_years:.1f} year{"s" if trading_years != 1 else ""})
- Number of simulations run: {n_sims:,}
- Model preset: {preset_name}
- EWMA decay factor (λ): {ewma_lambda}
- Student-t degrees of freedom: {t_dof}

PRICE FORECAST (based on {n_sims:,} simulated futures):
- Current price (starting point): ${S0:.2f}
- Most likely outcome (median): ${p50:.2f} ({(p50/S0-1)*100:+.1f}%)
- Optimistic scenario (95th percentile): ${p95:.2f} ({(p95/S0-1)*100:+.1f}%)
- Pessimistic scenario (5th percentile): ${p5:.2f} ({(p5/S0-1)*100:+.1f}%)
- Middle range (25th–75th percentile): ${p25:.2f} to ${p75:.2f}
- Best case across all simulations: ${best:.2f} ({(best/S0-1)*100:+.1f}%)
- Worst case across all simulations: ${worst:.2f} ({(worst/S0-1)*100:+.1f}%)
- Probability the stock is higher than today at end of forecast: {prob_gain:.1f}%

RISK METRICS:
- VaR 95% (maximum expected loss 95% of the time): ${var95:.2f} ({var95/S0*100:.1f}% of investment)
- CVaR 95% (average loss in the worst 5% of scenarios): ${cvar95:.2f} ({cvar95/S0*100:.1f}% of investment)
- Probability of losing more than 10%: {prob_loss10:.1f}%
- Probability of losing more than 25%: {prob_loss25:.1f}%

MODEL CHARACTERISTICS:
- Volatility regime: {vol_regime}
- Recent price drift: {drift_desc}
- Outcome spread (5th to 95th percentile): {spread_desc} ({spread_pct:.0f}% of starting price)

YOUR TASK:
Write a clear, friendly, and honest summary of these results for a retail investor. Structure your response with these four sections, each with a short bold heading:

1. **What the simulation is telling you** — Summarise the overall picture in 2–3 sentences. Is the model broadly optimistic, bearish, or uncertain? What does the median outcome mean in practical terms?

2. **The range of outcomes** — Explain the spread between the optimistic and pessimistic scenarios in plain English. Help the reader understand what "5th percentile" and "95th percentile" actually mean in real-world terms (e.g. "in roughly 1 in 20 simulations, the stock ended below $X").

3. **The risk picture** — Explain the downside risk in one or two sentences without using "VaR" or "CVaR" — describe what the numbers mean in plain terms. Mention the loss probabilities and what they imply for someone holding this stock.

4. **Important things to keep in mind** — Briefly note 2–3 honest limitations of this model (e.g. it's based on past data, it doesn't know about upcoming earnings or news, the median is not a prediction). End with a one-sentence disclaimer that this is not financial advice.

Keep the total response under 420 words. Do not use bullet points — write in flowing paragraphs. Do not repeat raw numbers excessively — reference them once to anchor the point, then speak to what they mean in human terms."""

        return prompt

    def show_ai_summary(ticker, info, params, paths, pcts, days,
                        n_sims, preset_name, ewma_lambda, t_dof, start, end):
        """Call Groq API and stream the AI summary into the UI."""
        st.subheader("🤖 AI Summary — Plain English Breakdown")
        st.markdown(
            "An AI-generated explanation of your simulation results, "
            "written for everyday investors."
        )

        groq_key = st.secrets.get("GROQ_API_KEY", "")
        if not groq_key:
            st.warning(
                "Groq API key not configured. Add `GROQ_API_KEY` to your Streamlit secrets "
                "to enable AI summaries. See deployment instructions for details."
            )
            return

        prompt = build_ai_prompt(
            ticker, info, params, paths, pcts, days,
            n_sims, preset_name, ewma_lambda, t_dof, start, end,
        )

        try:
            client = Groq(api_key=groq_key)
            with st.spinner("Generating AI summary…"):
                stream = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a clear, honest, and friendly financial educator. "
                                "You explain quantitative results in plain English for retail investors. "
                                "You never make price predictions or give investment advice. "
                                "You always remind users that simulations are based on historical data "
                                "and are not guarantees of future performance."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=650,
                    stream=True,
                )

                # Stream text into a styled container with blinking cursor
                summary_box = st.empty()
                full_text   = ""
                for chunk in stream:
                    delta      = chunk.choices[0].delta.content or ""
                    full_text += delta
                    summary_box.markdown(
                        f"""<div style="
                            background: #1a1a2e;
                            border-left: 4px solid #378ADD;
                            border-radius: 6px;
                            padding: 1.2rem 1.5rem;
                            color: #e0e0e0;
                            font-size: 0.97rem;
                            line-height: 1.75;
                            white-space: pre-wrap;
                        ">{full_text}▌</div>""",
                        unsafe_allow_html=True,
                    )

                # Final render — remove blinking cursor
                summary_box.markdown(
                    f"""<div style="
                        background: #1a1a2e;
                        border-left: 4px solid #378ADD;
                        border-radius: 6px;
                        padding: 1.2rem 1.5rem;
                        color: #e0e0e0;
                        font-size: 0.97rem;
                        line-height: 1.75;
                        white-space: pre-wrap;
                    ">{full_text}</div>""",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"AI summary unavailable: {e}")

    # ── Downloads ─────────────────────────────────────────────────────────────

    def build_csv(paths, pcts, S0, ticker, days, start, end, params,
                  ewma_lambda, t_dof, preset_name) -> bytes:
        final       = paths[:, -1]
        pnl_pct     = (final / S0 - 1) * 100
        p5, p25, p50, p75, p95 = (np.percentile(final, p) for p in [5, 25, 50, 75, 95])
        prob_profit = np.mean(final > S0) * 100

        summary_rows = [
            ["=== SIMULATION SUMMARY ===", ""],
            ["Ticker",                  ticker],
            ["Historical start",        str(start)],
            ["Historical end",          str(end)],
            ["Trading days in range",   params["num_hist_days"]],
            ["Days simulated forward",  days],
            ["Number of simulations",   paths.shape[0]],
            ["Preset used",             preset_name],
            ["EWMA lambda",             ewma_lambda],
            ["Student-t dof",           t_dof],
            ["Drift window (days)",     params["drift_window_used"]],
            ["", ""],
            ["Starting price (S0)",     f"${S0:.4f}"],
            ["Median forecast",         f"${p50:.4f}"],
            ["Median vs S0",            f"{(p50/S0-1)*100:+.2f}%"],
            ["Probability of gain",     f"{prob_profit:.1f}%"],
            ["EWMA vol (ann.)",         f"{params['ann_vol_ewma']*100:.2f}%"],
            ["Simple hist vol (ann.)",  f"{params['ann_vol_simple']*100:.2f}%"],
            ["Recent drift (ann.)",     f"{params['ann_drift_recent']*100:+.2f}%"],
            ["Full drift (ann.)",       f"{params['ann_drift_full']*100:+.2f}%"],
            ["", ""],
            ["=== PERCENTILE TABLE ===", ""],
            ["Percentile", "Price (USD)", "vs S0 (%)"],
            ["5th",           f"${p5:.4f}",            f"{(p5/S0-1)*100:+.2f}%"],
            ["25th",          f"${p25:.4f}",           f"{(p25/S0-1)*100:+.2f}%"],
            ["50th (median)", f"${p50:.4f}",           f"{(p50/S0-1)*100:+.2f}%"],
            ["75th",          f"${p75:.4f}",           f"{(p75/S0-1)*100:+.2f}%"],
            ["95th",          f"${p95:.4f}",           f"{(p95/S0-1)*100:+.2f}%"],
            ["Worst",         f"${np.min(final):.4f}", f"{(np.min(final)/S0-1)*100:+.2f}%"],
            ["Best",          f"${np.max(final):.4f}", f"{(np.max(final)/S0-1)*100:+.2f}%"],
            ["", ""],
            ["=== RISK METRICS ===", ""],
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
                f"${var_usd:.4f}",
                f"{var_pct:.2f}%",
                f"${cvar_usd:.4f}",
                f"{cvar_pct:.2f}%",
            ])

        buf    = io.StringIO()
        writer = csv.writer(buf)
        for row in summary_rows:
            writer.writerow(row)
        return buf.getvalue().encode()

    def fig_to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.read()

    def show_downloads(fig, paths, pcts, S0, ticker, days,
                       start, end, params, ewma_lambda, t_dof, preset_name):
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download chart (PNG)",
                data=fig_to_png(fig),
                file_name=f"{ticker}_montecarlo_{date.today()}.png",
                mime="image/png",
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

    # ── Plot ──────────────────────────────────────────────────────────────────

    def build_figure(ticker, closes, params, paths, pcts, days, n_sims, info, start, end):
        PANEL_BG = "#1a1a1a"
        TEXT     = "#e0e0e0"
        MUTED    = "#777777"
        BLUE     = "#378ADD"
        GREEN    = "#639922"
        RED      = "#D85A30"
        AMBER    = "#BA7517"
        HIST_COL = "#1c3a5e"
        FORE_COL = "#4caf78"

        fig = plt.figure(figsize=(15, 14))
        fig.patch.set_facecolor("#0f0f0f")

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[1.1, 1, 1.2],
            hspace=0.52, wspace=0.32,
        )
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.tick_params(which="minor", colors=MUTED, labelsize=0)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        S0     = params["last_price"]
        days_x = np.arange(days + 1)
        name   = info.get("longName", ticker.upper())

        # Panel 1 — simulation fan
        sample = min(n_sims, 120)
        idx    = np.random.choice(n_sims, sample, replace=False)
        for i in idx:
            ax1.plot(days_x, paths[i], color="#ffffff", alpha=0.04, linewidth=0.6)
        ax1.fill_between(days_x, pcts["p5"],  pcts["p95"], color=BLUE, alpha=0.12, label="5-95th %ile band")
        ax1.fill_between(days_x, pcts["p25"], pcts["p75"], color=BLUE, alpha=0.20, label="25-75th %ile band")
        ax1.plot(days_x, pcts["p95"], color=GREEN, linewidth=1.2, linestyle="--", label="95th percentile")
        ax1.plot(days_x, pcts["p50"], color=BLUE,  linewidth=2.0,                 label="Median")
        ax1.plot(days_x, pcts["p5"],  color=RED,   linewidth=1.2, linestyle="--", label="5th percentile")
        ax1.axhline(S0, color=AMBER, linewidth=1.0, linestyle=":", label=f"Starting price  ${S0:.2f}")
        apply_grid(ax1)
        ax1.set_title(
            f"{name} ({ticker.upper()}) — {n_sims:,} paths, {days} trading days forward\n"
            f"Historical window: {start}  →  {end}  ({params['num_hist_days']} trading days)  |  "
            f"EWMA vol: {params['ann_vol_ewma']:.1%}  |  "
            f"Recent drift: {params['ann_drift_recent']:+.1%}",
            color=TEXT, fontsize=10, pad=10,
        )
        ax1.set_xlabel("Trading days forward", color=MUTED, fontsize=9)
        ax1.set_ylabel("Simulated price (USD)", color=MUTED, fontsize=9)
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.3,
                   labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

        # Panel 2 — final price histogram
        final = paths[:, -1]
        ax2.hist(final, bins=50, color=BLUE, alpha=0.7, edgecolor=PANEL_BG, linewidth=0.3)
        ax2.axvline(np.percentile(final,  5), color=RED,   linestyle="--", linewidth=1.2, label="5th %ile")
        ax2.axvline(np.percentile(final, 50), color=BLUE,  linestyle="-",  linewidth=1.8, label="Median")
        ax2.axvline(np.percentile(final, 95), color=GREEN, linestyle="--", linewidth=1.2, label="95th %ile")
        ax2.axvline(S0, color=AMBER, linestyle=":", linewidth=1.2, label="Starting price")
        apply_grid(ax2)
        ax2.set_title(f"Final price distribution (day {days})", color=TEXT, fontsize=10, pad=8)
        ax2.set_xlabel("Price (USD)", color=MUTED, fontsize=9)
        ax2.set_ylabel("Number of simulations", color=MUTED, fontsize=9)
        ax2.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

        # Panel 3 — historical log-returns distribution
        rets   = params["log_returns"] * 100
        mean_r = rets.mean()
        std_r  = rets.std()
        ax3.hist(rets, bins=50, color=AMBER, alpha=0.7, edgecolor=PANEL_BG, linewidth=0.3)
        ax3.axvline(0,              color=MUTED, linewidth=0.8, linestyle="--")
        ax3.axvline(mean_r,         color=GREEN, linewidth=1.2, linestyle="--", label=f"Mean {mean_r:+.2f}%")
        ax3.axvline(mean_r - std_r, color=RED,   linewidth=1.0, linestyle=":",  label=f"±1 sigma  {std_r:.2f}%")
        ax3.axvline(mean_r + std_r, color=RED,   linewidth=1.0, linestyle=":")
        apply_grid(ax3)
        ax3.set_title(f"Historical daily log-returns\n{start}  →  {end}", color=TEXT, fontsize=10, pad=8)
        ax3.set_xlabel("Daily return (%)", color=MUTED, fontsize=9)
        ax3.set_ylabel("Frequency", color=MUTED, fontsize=9)
        ax3.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

        # Panel 4 — history + most-likely forecast
        hist_dates         = [(d.date() if hasattr(d, "date") else d) for d in closes.index]
        forecast_dates     = add_trading_days(hist_dates[-1], days)
        median_path        = find_median_path(paths, pcts)
        forecast_end_price = median_path[-1]
        pct_change         = (forecast_end_price / S0 - 1) * 100

        def to_mpl(d):
            if isinstance(d, datetime):
                return mdates.date2num(d)
            return mdates.date2num(datetime(d.year, d.month, d.day))

        hist_x     = [to_mpl(d) for d in hist_dates]
        forecast_x = [to_mpl(d) for d in forecast_dates]
        bridge_x   = [hist_x[-1]] + forecast_x
        bridge_y   = median_path
        conf_low   = np.concatenate([[S0], pcts["p5"][1:]])
        conf_high  = np.concatenate([[S0], pcts["p95"][1:]])

        ax4.plot(hist_x, closes.values, color=HIST_COL, linewidth=1.8, label="History", zorder=3)
        ax4.plot(bridge_x, bridge_y, color=FORE_COL, linewidth=1.8,
                 label=f"Most-likely forecast ({days}d)", zorder=3)
        ax4.fill_between(bridge_x, conf_low, conf_high,
                         color=FORE_COL, alpha=0.12, label="5-95th %ile band")
        divider_x = hist_x[-1]
        ax4.axvline(divider_x, color=MUTED, linewidth=1.0, linestyle="--", zorder=4)
        ax4.xaxis_date()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        for label in ax4.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        apply_grid(ax4, date_axis=True)

        all_y = list(closes.values) + list(median_path) + list(conf_low) + list(conf_high)
        y_min = min(all_y) * 0.97
        y_max = max(all_y) * 1.03
        ax4.set_ylim(y_min, y_max)

        ax4.text(
            divider_x + (forecast_x[-1] - hist_x[-1]) * 0.01,
            y_max * 0.98, "  Forecast start",
            color=MUTED, fontsize=8, va="top", zorder=5,
        )
        ax4.annotate(
            f"  ${S0:.2f}",
            xy=(hist_x[-1], S0), xytext=(6, 0), textcoords="offset points",
            color=TEXT, fontsize=8, va="center", zorder=5,
        )
        ax4.annotate(
            f"${forecast_end_price:.2f}  ({pct_change:+.1f}%)",
            xy=(forecast_x[-1], forecast_end_price),
            xytext=(-88, 10), textcoords="offset points",
            color=FORE_COL, fontsize=8, fontweight="bold", zorder=5,
            arrowprops=dict(arrowstyle="-", color=FORE_COL, lw=0.8),
        )
        ax4.set_title(
            f"{name} ({ticker.upper()}) — History & Most-Likely Forecast\n"
            f"History: {hist_dates[0]} → {hist_dates[-1]}  |  "
            f"Forecast: {forecast_dates[0]} → {forecast_dates[-1]}  ({days} trading days)",
            color=TEXT, fontsize=10, pad=10,
        )
        ax4.set_xlabel("Date", color=MUTED, fontsize=9)
        ax4.set_ylabel("Price (USD)", color=MUTED, fontsize=9)
        ax4.legend(loc="upper left", fontsize=8, framealpha=0.3,
                   labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

        plt.suptitle(
            f"Monte Carlo Simulation — {ticker.upper()}  |  "
            f"EWMA vol: {params['ann_vol_scaled']:.1%} ann.  |  "
            f"Recent drift: {params['ann_drift_recent']:+.1%} ann.  |  "
            f"Data: {start} → {end}",
            color=TEXT, fontsize=11, y=0.99,
        )
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
                    closes,
                    drift_override=drift_override,
                    vol_scale=vol_scale,
                    ewma_lambda=ewma_lambda,
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
                show_ai_summary(
                    ticker, info, params, paths, pcts, days_forward,
                    num_simulations, preset_name, ewma_lambda, t_dof,
                    start_date, end_date,
                )

                st.divider()
                st.subheader("📉 Charts")
                fig = build_figure(
                    ticker, closes, params, paths, pcts,
                    days_forward, num_simulations, info,
                    start_date, end_date,
                )
                st.pyplot(fig, use_container_width=True)

                st.divider()
                show_downloads(
                    fig, paths, pcts, S0, ticker, days_forward,
                    start_date, end_date, params, ewma_lambda, t_dof, preset_name,
                )
                plt.close(fig)

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    else:
        st.info("👈 Configure your parameters in the sidebar, then click **▶ Run Simulation**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT & METHODOLOGY
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
The date range you choose determines how much history is used to calibrate the model.

### 2. Compute log returns
Daily log returns are calculated as:

> **r_t = ln(P_t / P_{t-1})**

Log returns are preferred over simple returns because they are additive across time
and more closely approximate a normal distribution — a core assumption of GBM.

### 3. Estimate volatility — EWMA
Rather than treating all historical returns equally, this model uses
**Exponentially Weighted Moving Average (EWMA)** volatility:

> **σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_t**

The decay factor **λ** controls how quickly older observations lose influence.
The industry standard (JP Morgan RiskMetrics) is **λ = 0.94**.
A lower λ makes the model react faster to recent volatility spikes; a higher λ
smooths out short-term noise.

### 4. Estimate drift — recent window
The expected daily drift **μ** is estimated from the mean of log returns over a
recent window (default: last 63 trading days ≈ 3 months), rather than the full
historical period. This makes the model sensitive to current momentum rather than
anchoring to where the stock was years ago.

### 5. Simulate price paths — GBM with fat tails
Each simulation path is generated day-by-day using the GBM formula:

> **S_{t+1} = S_t · exp( (μ − ½σ²) + σ · Z_t )**

where **Z_t** is a random shock drawn from a **Student's t-distribution** with
user-configurable degrees of freedom (default: 5). The t-distribution has heavier
tails than the normal — meaning extreme daily moves occur more often, as they do
in real markets.

### 6. Aggregate results
After running all simulations, the model computes percentile bands, a most-likely
path, VaR/CVaR risk metrics, and feeds everything to the AI summary engine.

---

## AI Summary
After each simulation, results are sent to **Llama 3 70B** (via Groq) which generates
a plain-English explanation tailored for retail investors — covering the forecast
outlook, range of outcomes, risk picture, and model limitations.

---

## Risk metrics explained

| Metric | What it answers |
|---|---|
| **VaR (Value at Risk)** | "What is the worst loss I should expect NOT to exceed, X% of the time?" |
| **CVaR / Expected Shortfall** | "If I end up in the worst X% of outcomes, what is my average loss?" |

CVaR is considered a more complete risk measure than VaR because it captures the
*severity* of tail events, not just their threshold.

---

## Preset configurations

| Preset | Best for | Key settings |
|---|---|---|
| **Conservative** | Large-cap, low-beta stocks (JNJ, KO, BRK) | Low vol scale, slow EWMA decay, near-normal tails, long drift window |
| **Balanced** | Most equities — a solid default | RiskMetrics λ=0.94, t-dof=5, 63-day drift window |
| **Aggressive** | High-beta / speculative names (TSLA, MSTR) | 1.5× vol, fast EWMA decay, very fat tails (dof=3) |
| **Momentum** | Stocks in a strong near-term trend | Standard vol, 21-day drift window to capture recent move |

---

## Limitations & important disclaimers

- **GBM assumes constant volatility and drift.** Real markets exhibit volatility clustering, mean reversion, jumps, and regime changes that this model does not fully capture.
- **Past price behaviour does not guarantee future results.**
- **The "most-likely forecast" line is the median of a distribution, not a prediction.** By construction, roughly half of all simulated paths end above it and half below.
- **VaR and CVaR are model-derived estimates**, not guarantees.
- This tool is intended for **educational and analytical purposes only** and does not constitute financial advice. Always consult a qualified financial professional before making investment decisions.

---

## Technology

Built with **Python**, **Streamlit**, **yfinance**, **NumPy**, **SciPy**, **Matplotlib**, and **Groq (Llama 3 70B)**.
    """)