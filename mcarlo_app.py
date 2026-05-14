"""
monte carlo stock price simulation - streamlit app
====================================================
all data derived from yfinance and cached for performance 
run with:
    streamlit run mcarlo_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
import streamlit as st
from datetime import date, datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Price Action Forecast via Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Price Action Forecast via Monte Carlo Simulation")
st.markdown(
    "Fetches real historical data via **yfinance** and simulates future price "
    "paths using **Geometric Brownian Motion (GBM)**."
)

# ── Sidebar — all user inputs ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    ticker = st.text_input(
        "Ticker symbol",
        value="AAPL",
        help="e.g. AAPL, TSLA, NVDA, SPY",
    ).strip().upper()

    st.subheader("Historical data window")
    today   = date.today()
    one_yr  = today - timedelta(days=365)

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
        min_value=21,
        max_value=756,
        value=252,
        step=21,
        help="252 ≈ 1 year of trading days.",
    )
    num_simulations = st.slider(
        "Number of simulations",
        min_value=100,
        max_value=10000,
        value=500,
        step=100,
    )

    st.subheader("Advanced")
    vol_scale = st.slider(
        "Volatility multiplier",
        min_value=0.25,
        max_value=3.0,
        value=1.0,
        step=0.25,
        help="1.0 = historical vol. 2.0 = double, 0.5 = half.",
    )
    use_drift_override = st.checkbox("Override annual drift?", value=False)
    drift_override = None
    if use_drift_override:
        drift_override = st.number_input(
            "Annual drift (e.g. 0.10 = +10%)",
            min_value=-1.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            format="%.2f",
        )

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Helper functions (identical logic to mcarlo.py) ───────────────────────────

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


def compute_parameters(closes: pd.Series, drift_override=None, vol_scale=1.0) -> dict:
    log_returns   = np.log(closes / closes.shift(1)).dropna()
    daily_mean    = log_returns.mean()
    daily_std     = log_returns.std()
    ann_drift     = daily_mean * 252
    ann_vol       = daily_std  * np.sqrt(252)
    if drift_override is not None:
        daily_mean = drift_override / 252
    daily_std_scaled = daily_std * vol_scale
    return {
        "daily_mean":     daily_mean,
        "daily_std":      daily_std_scaled,
        "ann_drift":      ann_drift,
        "ann_vol":        ann_vol,
        "ann_vol_scaled": daily_std_scaled * np.sqrt(252),
        "log_returns":    log_returns,
        "last_price":     float(closes.iloc[-1]),
        "num_hist_days":  len(closes),
    }


def run_simulation(params: dict, days: int, n_sims: int) -> np.ndarray:
    mu, sigma, S0 = params["daily_mean"], params["daily_std"], params["last_price"]
    np.random.seed(42)
    Z             = np.random.standard_normal((n_sims, days))
    daily_returns = (mu - 0.5 * sigma**2) + sigma * Z
    log_paths     = np.concatenate([np.zeros((n_sims, 1)), daily_returns], axis=1)
    return S0 * np.exp(np.cumsum(log_paths, axis=1))


def compute_percentiles(paths: np.ndarray) -> dict:
    return {
        "p5":  np.percentile(paths,  5, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }


def find_median_path(paths: np.ndarray, pcts: dict) -> np.ndarray:
    median_final = pcts["p50"][-1]
    closest_idx  = np.argmin(np.abs(paths[:, -1] - median_final))
    return paths[closest_idx]


# ── Summary metrics (replaces print_summary) ──────────────────────────────────

def show_summary(ticker, params, paths, days, start, end):
    final = paths[:, -1]
    S0    = params["last_price"]
    p5, p25, p50, p75, p95 = (np.percentile(final, p) for p in [5, 25, 50, 75, 95])
    prob_profit = np.mean(final > S0) * 100
    worst, best = np.min(final), np.max(final)

    st.subheader("📊 Simulation Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Starting price (S0)",   f"${S0:.2f}")
    col2.metric("Median forecast",       f"${p50:.2f}",  f"{(p50/S0-1):+.1%}")
    col3.metric("Probability of gain",   f"{prob_profit:.1f}%")
    col4.metric("Historical volatility", f"{params['ann_vol']:.1%} ann.")

    with st.expander("Full percentile table"):
        summary_df = pd.DataFrame({
            "Percentile": ["5th", "25th", "50th (median)", "75th", "95th", "Worst path", "Best path"],
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
| Historical drift (ann.) | {params['ann_drift']:+.1%} |
| Historical volatility (ann.) | {params['ann_vol']:.1%} |
| Simulated volatility (ann., scaled) | {params['ann_vol_scaled']:.1%} |
| Days simulated forward | {days} |
        """)


# ── Plot (identical logic to mcarlo.py, returns fig instead of plt.show()) ────

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

    # Panel 1: all paths
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
        f"Historical window: {start}  →  {end}  ({params['num_hist_days']} trading days)",
        color=TEXT, fontsize=10, pad=10,
    )
    ax1.set_xlabel("Trading days forward", color=MUTED, fontsize=9)
    ax1.set_ylabel("Simulated price (USD)", color=MUTED, fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3,
               labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

    # Panel 2: final price histogram
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

    # Panel 3: historical log-returns
    rets   = params["log_returns"] * 100
    mean_r = rets.mean()
    std_r  = rets.std()
    ax3.hist(rets, bins=50, color=AMBER, alpha=0.7, edgecolor=PANEL_BG, linewidth=0.3)
    ax3.axvline(0,              color=MUTED, linewidth=0.8, linestyle="--")
    ax3.axvline(mean_r,         color=GREEN, linewidth=1.2, linestyle="--", label=f"Mean {mean_r:+.2f}%")
    ax3.axvline(mean_r - std_r, color=RED,   linewidth=1.0, linestyle=":",  label=f"±1 sigma  {std_r:.2f}%")
    ax3.axvline(mean_r + std_r, color=RED,   linewidth=1.0, linestyle=":")
    apply_grid(ax3)
    ax3.set_title(
        f"Historical daily log-returns\n{start}  →  {end}",
        color=TEXT, fontsize=10, pad=8,
    )
    ax3.set_xlabel("Daily return (%)", color=MUTED, fontsize=9)
    ax3.set_ylabel("Frequency", color=MUTED, fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor="#444")

    # Panel 4: history + most-likely forecast
    hist_dates = [(d.date() if hasattr(d, "date") else d) for d in closes.index]
    forecast_dates = add_trading_days(hist_dates[-1], days)
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
        f"Vol: {params['ann_vol_scaled']:.1%} ann.  |  "
        f"Drift: {params['ann_drift']:+.1%} ann.  |  "
        f"Data: {start} → {end}",
        color=TEXT, fontsize=11, y=0.99,
    )

    return fig


# ── Main app body ─────────────────────────────────────────────────────────────

if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    elif (end_date - start_date).days < 30:
        st.error("Date range must span at least 30 calendar days.")
    else:
        try:
            closes, info = fetch_data(ticker, start_date, end_date)
            params       = compute_parameters(closes, drift_override, vol_scale)

            with st.spinner("Running simulations…"):
                paths = run_simulation(params, days_forward, num_simulations)

            pcts = compute_percentiles(paths)

            show_summary(ticker, params, paths, days_forward, start_date, end_date)

            st.subheader("📉 Charts")
            fig = build_figure(
                ticker, closes, params, paths, pcts,
                days_forward, num_simulations, info,
                start_date, end_date,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

else:
    st.info("👈 Configure your parameters in the sidebar, then click **▶ Run Simulation**.")
