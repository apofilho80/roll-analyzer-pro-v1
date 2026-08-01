
import math
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Professional Roll Analyzer",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 14px;
            padding: 12px;
        }
        .small-note {font-size: 0.88rem; opacity: 0.75;}
        .good {color:#1a7f37; font-weight:700;}
        .warn {color:#b7791f; font-weight:700;}
        .bad {color:#c62828; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA MODELS
# ============================================================
@dataclass
class VerticalPutSpread:
    ticker: str
    expiration: date
    short_strike: float
    long_strike: float
    quantity: int
    entry_credit: float   # per share
    current_close_debit: float = 0.0  # per share, optional

    @property
    def width(self) -> float:
        return self.short_strike - self.long_strike

    @property
    def max_profit(self) -> float:
        return self.entry_credit * 100 * self.quantity

    @property
    def max_loss(self) -> float:
        return max(self.width - self.entry_credit, 0) * 100 * self.quantity

    @property
    def breakeven(self) -> float:
        return self.short_strike - self.entry_credit


@dataclass
class RollCandidate:
    name: str
    expiration: date
    short_strike: float
    long_strike: float
    roll_credit: float  # net credit for entire roll, per share
    quantity: int

    @property
    def width(self) -> float:
        return self.short_strike - self.long_strike


# ============================================================
# HELPERS
# ============================================================
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def fetch_spot(ticker: str) -> Optional[float]:
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probability_above(
    spot: float,
    target: float,
    days: int,
    iv: float,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Lognormal approximation to probability that price finishes above target.
    IV must be entered as a decimal, e.g. 0.35 for 35%.
    """
    if spot <= 0 or target <= 0:
        return 0.0
    if days <= 0:
        return 1.0 if spot > target else 0.0
    if iv <= 0:
        return 1.0 if spot > target else 0.0

    t = days / 365.0
    z = (
        math.log(spot / target)
        + (risk_free_rate - 0.5 * iv * iv) * t
    ) / (iv * math.sqrt(t))
    return normal_cdf(z)


def candidate_metrics(
    current: VerticalPutSpread,
    candidate: RollCandidate,
    spot: float,
    iv: float,
    risk_free_rate: float,
    max_allowed_risk_increase: float,
) -> dict:
    total_credit = current.entry_credit + candidate.roll_credit
    new_breakeven = candidate.short_strike - total_credit
    new_width = candidate.width
    new_max_profit = total_credit * 100 * candidate.quantity
    new_max_loss = max(new_width - total_credit, 0) * 100 * candidate.quantity

    days_new = max((candidate.expiration - date.today()).days, 0)
    pop = probability_above(
        spot=spot,
        target=new_breakeven,
        days=days_new,
        iv=iv,
        risk_free_rate=risk_free_rate,
    )

    be_improvement = current.breakeven - new_breakeven
    strike_improvement = current.short_strike - candidate.short_strike
    risk_change = new_max_loss - current.max_loss

    # Return on risk based on total premium accumulated.
    ror = new_max_profit / new_max_loss if new_max_loss > 0 else 0.0
    annualized_ror = (
        (ror * 365.0 / days_new) if days_new > 0 else 0.0
    )

    # --------------------------------------------------------
    # SCORE, 0-100
    # --------------------------------------------------------
    score = 50.0

    # Probability of profit: up to +20 / -15
    score += np.interp(pop, [0.35, 0.60, 0.80, 0.95], [-15, 0, 12, 20])

    # Breakeven improvement: up to +18
    score += np.interp(be_improvement, [-25, 0, 15, 40], [-15, 0, 9, 18])

    # Lowering short strike: up to +10
    score += np.interp(strike_improvement, [-20, 0, 15, 40], [-10, 0, 5, 10])

    # Credit quality: up to +12
    credit_pct_width = candidate.roll_credit / max(candidate.width, 0.01)
    score += np.interp(credit_pct_width, [-0.05, 0, 0.10, 0.25], [-8, 0, 6, 12])

    # Risk change: penalize excessive max-loss expansion
    risk_ratio = risk_change / max(current.max_loss, 1.0)
    allowed_ratio = max_allowed_risk_increase / 100.0
    if risk_ratio <= 0:
        score += 10
    elif risk_ratio <= allowed_ratio:
        score += np.interp(risk_ratio, [0, allowed_ratio], [8, 0])
    else:
        score -= np.interp(
            risk_ratio,
            [allowed_ratio, max(allowed_ratio + 0.01, 2.5)],
            [5, 25],
        )

    # Width expansion penalty
    width_ratio = new_width / max(current.width, 0.01)
    score -= np.interp(width_ratio, [1.0, 1.5, 2.0, 3.0], [0, 3, 10, 20])

    score = float(np.clip(score, 0, 100))

    if score >= 85:
        verdict = "Strong candidate"
    elif score >= 70:
        verdict = "Reasonable candidate"
    elif score >= 55:
        verdict = "Borderline"
    else:
        verdict = "Avoid / redesign"

    return {
        "Candidate": candidate.name,
        "Expiration": candidate.expiration.isoformat(),
        "Short Put": candidate.short_strike,
        "Long Put": candidate.long_strike,
        "Width": new_width,
        "Roll Credit / Share": candidate.roll_credit,
        "Total Credit / Share": total_credit,
        "New Break-even": new_breakeven,
        "BE Improvement": be_improvement,
        "Short-Strike Improvement": strike_improvement,
        "Max Profit": new_max_profit,
        "Max Loss": new_max_loss,
        "Risk Change": risk_change,
        "POP Proxy": pop,
        "Return on Risk": ror,
        "Annualized ROR Proxy": annualized_ror,
        "Score": score,
        "Verdict": verdict,
    }


def payoff_at_expiration(
    stock_prices: np.ndarray,
    short_strike: float,
    long_strike: float,
    total_credit: float,
    quantity: int,
) -> np.ndarray:
    short_put = -np.maximum(short_strike - stock_prices, 0.0)
    long_put = np.maximum(long_strike - stock_prices, 0.0)
    return (short_put + long_put + total_credit) * 100 * quantity


# ============================================================
# HEADER
# ============================================================
st.title("📊 Professional Roll Analyzer")
st.caption(
    "Compare defined-risk bull-put-spread rolls using break-even improvement, "
    "maximum risk, probability proxy, return on risk, and a customizable score."
)

with st.expander("Important assumptions", expanded=False):
    st.markdown(
        """
        - This V1 analyzes **bull put spread → bull put spread** rolls.
        - Credits are entered **per share**. One option contract represents 100 shares.
        - The probability-of-profit figure is a simplified lognormal proxy, not a broker POP.
        - Taxes, commissions, early assignment, dividends, volatility skew, and liquidity are not fully modeled.
        - A high score does not make a trade safe. The score is a comparison tool.
        """
    )


# ============================================================
# SIDEBAR — MARKET ASSUMPTIONS
# ============================================================
st.sidebar.header("Market assumptions")

ticker = st.sidebar.text_input("Ticker", value="META").upper().strip()
use_live_spot = st.sidebar.checkbox("Try Yahoo Finance spot price", value=True)

manual_spot = st.sidebar.number_input(
    "Manual stock price",
    min_value=0.01,
    value=556.71,
    step=1.0,
)

spot = manual_spot
if use_live_spot:
    live = fetch_spot(ticker)
    if live is not None:
        spot = live
        st.sidebar.success(f"Spot loaded: ${spot:,.2f}")
    else:
        st.sidebar.warning("Live price unavailable; using manual price.")

iv_pct = st.sidebar.number_input(
    "Implied volatility assumption (%)",
    min_value=1.0,
    max_value=300.0,
    value=35.0,
    step=1.0,
)
iv = iv_pct / 100.0

risk_free_pct = st.sidebar.number_input(
    "Risk-free rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.25,
)
risk_free_rate = risk_free_pct / 100.0

max_risk_increase_pct = st.sidebar.slider(
    "Maximum acceptable max-loss increase",
    min_value=0,
    max_value=200,
    value=50,
    step=5,
    format="%d%%",
)


# ============================================================
# CURRENT POSITION
# ============================================================
st.header("1. Current position")

c1, c2, c3, c4 = st.columns(4)
with c1:
    current_exp = st.date_input(
        "Current expiration",
        value=date(2026, 8, 21),
        key="current_exp",
    )
with c2:
    current_short = st.number_input(
        "Current short-put strike",
        min_value=0.01,
        value=625.0,
        step=5.0,
    )
with c3:
    current_long = st.number_input(
        "Current long-put strike",
        min_value=0.01,
        value=580.0,
        step=5.0,
    )
with c4:
    quantity = st.number_input(
        "Number of spreads",
        min_value=1,
        value=2,
        step=1,
    )

c5, c6 = st.columns(2)
with c5:
    original_credit = st.number_input(
        "Original credit received per share",
        min_value=0.0,
        value=20.22,
        step=0.01,
    )
with c6:
    close_debit = st.number_input(
        "Current debit to close per share (optional)",
        min_value=0.0,
        value=0.0,
        step=0.05,
        help="Used for reference only in this V1. Enter the total debit needed to close the old spread.",
    )

current = VerticalPutSpread(
    ticker=ticker,
    expiration=current_exp,
    short_strike=current_short,
    long_strike=current_long,
    quantity=int(quantity),
    entry_credit=original_credit,
    current_close_debit=close_debit,
)

if current.long_strike >= current.short_strike:
    st.error("The long-put strike must be below the short-put strike.")
    st.stop()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stock price", f"${spot:,.2f}")
m2.metric("Current width", f"${current.width:,.2f}")
m3.metric("Current break-even", f"${current.breakeven:,.2f}")
m4.metric("Original max profit", f"${current.max_profit:,.0f}")
m5.metric("Current max loss", f"${current.max_loss:,.0f}")


# ============================================================
# CANDIDATES
# ============================================================
st.header("2. Roll candidates")
st.write(
    "Enter up to five candidate rolls. The roll credit is the **net credit for the complete roll**, per share."
)

default_candidates = [
    ("Dec 600/510", date(2026, 12, 18), 600.0, 510.0, 10.10),
    ("Dec 595/525", date(2026, 12, 18), 595.0, 525.0, 8.30),
    ("Jan 590/530", date(2027, 1, 15), 590.0, 530.0, 9.60),
]

candidate_count = st.slider("Number of candidates", 1, 5, 3)

candidates = []
for i in range(candidate_count):
    defaults = default_candidates[i] if i < len(default_candidates) else (
        f"Candidate {i+1}",
        date(2026, 12, 18),
        590.0,
        530.0,
        5.0,
    )

    with st.expander(f"Candidate {i+1}", expanded=(i == 0)):
        a, b, c, d, e = st.columns([1.4, 1.2, 1, 1, 1.2])
        with a:
            name = st.text_input(
                "Name",
                value=defaults[0],
                key=f"name_{i}",
            )
        with b:
            exp = st.date_input(
                "Expiration",
                value=defaults[1],
                key=f"exp_{i}",
            )
        with c:
            short = st.number_input(
                "Short put",
                min_value=0.01,
                value=defaults[2],
                step=5.0,
                key=f"short_{i}",
            )
        with d:
            long = st.number_input(
                "Long put",
                min_value=0.01,
                value=defaults[3],
                step=5.0,
                key=f"long_{i}",
            )
        with e:
            credit = st.number_input(
                "Net roll credit/share",
                value=defaults[4],
                step=0.05,
                key=f"credit_{i}",
            )

        if long >= short:
            st.warning("Long-put strike must be below short-put strike.")
        else:
            candidates.append(
                RollCandidate(
                    name=name,
                    expiration=exp,
                    short_strike=short,
                    long_strike=long,
                    roll_credit=credit,
                    quantity=int(quantity),
                )
            )


# ============================================================
# ANALYSIS
# ============================================================
st.header("3. Professional comparison")

if not candidates:
    st.warning("Add at least one valid candidate.")
    st.stop()

rows = [
    candidate_metrics(
        current=current,
        candidate=candidate,
        spot=spot,
        iv=iv,
        risk_free_rate=risk_free_rate,
        max_allowed_risk_increase=max_risk_increase_pct,
    )
    for candidate in candidates
]

df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

display_df = df.copy()
for col in [
    "Roll Credit / Share",
    "Total Credit / Share",
    "New Break-even",
    "BE Improvement",
    "Short-Strike Improvement",
    "Width",
]:
    display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}")

for col in ["Max Profit", "Max Loss", "Risk Change"]:
    display_df[col] = display_df[col].map(lambda x: f"${x:,.0f}")

for col in ["POP Proxy", "Return on Risk", "Annualized ROR Proxy"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")

display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.1f}")

st.dataframe(
    display_df[
        [
            "Candidate",
            "Expiration",
            "Short Put",
            "Long Put",
            "Width",
            "Roll Credit / Share",
            "Total Credit / Share",
            "New Break-even",
            "BE Improvement",
            "Max Loss",
            "Risk Change",
            "POP Proxy",
            "Annualized ROR Proxy",
            "Score",
            "Verdict",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

best = df.iloc[0]

st.subheader("Top-ranked candidate")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Candidate", best["Candidate"])
r2.metric("Professional score", f'{best["Score"]:.1f}/100')
r3.metric("New break-even", f'${best["New Break-even"]:,.2f}')
r4.metric("POP proxy", f'{best["POP Proxy"]:.1%}')

risk_delta = best["Risk Change"]
if risk_delta > 0:
    risk_text = f"increases maximum loss by ${risk_delta:,.0f}"
    risk_class = "bad"
else:
    risk_text = f"reduces maximum loss by ${abs(risk_delta):,.0f}"
    risk_class = "good"

st.markdown(
    f"""
    **Interpretation:** The highest-ranked roll is **{best["Candidate"]}**.
    It improves the break-even by **${best["BE Improvement"]:,.2f} per share**,
    lowers the short strike by **${best["Short-Strike Improvement"]:,.2f}**, and
    <span class="{risk_class}">{risk_text}</span>.
    """,
    unsafe_allow_html=True,
)

if risk_delta > current.max_loss * (max_risk_increase_pct / 100):
    st.error(
        "Risk warning: this candidate increases maximum loss beyond your selected tolerance."
    )
elif best["Score"] >= 85:
    st.success("The candidate passes the current scoring thresholds.")
elif best["Score"] >= 70:
    st.warning("The candidate is reasonable, but review the added risk and width.")
else:
    st.error("The highest-ranked candidate is still weak under the selected assumptions.")


# ============================================================
# PAYOFF CHART
# ============================================================
st.header("4. Expiration payoff comparison")

selected_name = st.selectbox(
    "Candidate for payoff chart",
    options=df["Candidate"].tolist(),
)

selected_row = df.loc[df["Candidate"] == selected_name].iloc[0]
selected_candidate = next(c for c in candidates if c.name == selected_name)

lower = max(1.0, min(current.long_strike, selected_candidate.long_strike, spot) * 0.70)
upper = max(current.short_strike, selected_candidate.short_strike, spot) * 1.25
prices = np.linspace(lower, upper, 300)

current_payoff = payoff_at_expiration(
    prices,
    current.short_strike,
    current.long_strike,
    current.entry_credit,
    current.quantity,
)

new_total_credit = current.entry_credit + selected_candidate.roll_credit
rolled_payoff = payoff_at_expiration(
    prices,
    selected_candidate.short_strike,
    selected_candidate.long_strike,
    new_total_credit,
    selected_candidate.quantity,
)

chart_df = pd.DataFrame(
    {
        "Stock price at expiration": prices,
        "Current position": current_payoff,
        "Rolled position": rolled_payoff,
    }
).set_index("Stock price at expiration")

st.line_chart(chart_df, use_container_width=True)

st.caption(
    "The rolled payoff includes the original credit plus the new net roll credit."
)


# ============================================================
# SCENARIO TABLE
# ============================================================
st.header("5. Scenario analysis")

scenario_prices = sorted(set([
    round(spot * 0.75, 2),
    round(spot * 0.85, 2),
    round(spot, 2),
    round(selected_candidate.long_strike, 2),
    round(selected_candidate.short_strike, 2),
    round(spot * 1.15, 2),
]))

scenario_rows = []
for s in scenario_prices:
    curr = payoff_at_expiration(
        np.array([s]),
        current.short_strike,
        current.long_strike,
        current.entry_credit,
        current.quantity,
    )[0]
    roll = payoff_at_expiration(
        np.array([s]),
        selected_candidate.short_strike,
        selected_candidate.long_strike,
        new_total_credit,
        selected_candidate.quantity,
    )[0]
    scenario_rows.append(
        {
            "META at expiration": s,
            "Current position P/L": curr,
            "Rolled position P/L": roll,
            "Roll advantage": roll - curr,
        }
    )

scenario_df = pd.DataFrame(scenario_rows)
st.dataframe(
    scenario_df.style.format(
        {
            "META at expiration": "${:,.2f}",
            "Current position P/L": "${:,.0f}",
            "Rolled position P/L": "${:,.0f}",
            "Roll advantage": "${:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# EXPORT
# ============================================================
st.header("6. Export")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download roll comparison CSV",
    data=csv,
    file_name=f"{ticker}_roll_analysis.csv",
    mime="text/csv",
)

st.markdown(
    '<div class="small-note">'
    'V1 is designed for manual entries so that it remains dependable even when an '
    'options-data API is unavailable. A later version can import option chains and '
    'automatically generate candidate rolls.'
    '</div>',
    unsafe_allow_html=True,
)
