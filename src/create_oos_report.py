
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "out_of_sample_summary.csv"
OUTPUT = ROOT / "reports" / "out_of_sample_dashboard.html"
CSV_OUTPUT = ROOT / "reports" / "out_of_sample_comparison_formatted.csv"

METRICS = [
    "Total Return",
    "CAGR",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Maximum Drawdown",
    "Calmar Ratio",
]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT}. Run validate_out_of_sample.py first."
        )

    df = pd.read_csv(INPUT)

    required = ["Strategy"] + METRICS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Percentages are displayed as percentages; ratios remain multiples.
    display = pd.DataFrame({
        "Strategy": df["Strategy"],
        "Total Return (%)": df["Total Return"] * 100,
        "CAGR (%)": df["CAGR"] * 100,
        "Annualized Volatility (%)": df["Annualized Volatility"] * 100,
        "Sharpe Ratio": df["Sharpe Ratio"],
        "Maximum Drawdown (%)": df["Maximum Drawdown"] * 100,
        "Calmar Ratio": df["Calmar Ratio"],
    })

    display.to_csv(CSV_OUTPUT, index=False, float_format="%.4f")

    # Choose a strategy for the headline KPIs based on the highest
    # Sharpe ratio in the supplied results, not on future performance.
    best_idx = df["Sharpe Ratio"].idxmax()
    best = df.loc[best_idx]

    test_start = (
        str(df["Test Start"].dropna().iloc[0])
        if "Test Start" in df.columns and df["Test Start"].notna().any()
        else "Not provided"
    )
    test_end = (
        str(df["Test End"].dropna().iloc[0])
        if "Test End" in df.columns and df["Test End"].notna().any()
        else "Not provided"
    )

    observations = (
        int(df["Observations"].min())
        if "Observations" in df.columns and df["Observations"].notna().any()
        else None
    )

    # Chart 1: returns.
    returns_fig = go.Figure()
    for metric, label in [
        ("Total Return", "Total return"),
        ("CAGR", "CAGR"),
    ]:
        returns_fig.add_trace(go.Bar(
            x=df["Strategy"],
            y=df[metric] * 100,
            name=label,
            text=[f"{v:.2f}%" for v in df[metric] * 100],
            textposition="outside",
            hovertemplate="%{x}<br>" + label + ": %{y:.2f}%<extra></extra>",
        ))

    returns_fig.update_layout(
        title="Returns Comparison",
        barmode="group",
        yaxis_title="Return (%)",
        template="plotly_white",
        legend_title="Metric",
        margin=dict(t=70, b=70, l=60, r=30),
    )

    # Chart 2: risk.
    risk_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Annualized Volatility", "Maximum Drawdown"),
    )

    risk_fig.add_trace(go.Bar(
        x=df["Strategy"],
        y=df["Annualized Volatility"] * 100,
        name="Volatility",
        text=[f"{v:.2f}%" for v in df["Annualized Volatility"] * 100],
        textposition="outside",
        hovertemplate="%{x}<br>Volatility: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)

    risk_fig.add_trace(go.Bar(
        x=df["Strategy"],
        y=df["Maximum Drawdown"] * 100,
        name="Maximum drawdown",
        text=[f"{v:.2f}%" for v in df["Maximum Drawdown"] * 100],
        textposition="outside",
        hovertemplate="%{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ), row=1, col=2)

    risk_fig.update_layout(
        title="Risk Comparison",
        template="plotly_white",
        showlegend=False,
        height=480,
        margin=dict(t=90, b=100, l=55, r=30),
    )
    risk_fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    risk_fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)

    # Chart 3: risk-adjusted performance.
    ratios_fig = go.Figure()
    for metric, label in [
        ("Sharpe Ratio", "Sharpe ratio"),
        ("Calmar Ratio", "Calmar ratio"),
    ]:
        ratios_fig.add_trace(go.Bar(
            x=df["Strategy"],
            y=df[metric],
            name=label,
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="outside",
            hovertemplate="%{x}<br>" + label + ": %{y:.3f}<extra></extra>",
        ))

    ratios_fig.update_layout(
        title="Risk-Adjusted Performance",
        barmode="group",
        yaxis_title="Ratio",
        template="plotly_white",
        legend_title="Metric",
        margin=dict(t=70, b=70, l=60, r=30),
    )

    # Detailed comparison table.
    formatted = display.copy()
    for col in [
        "Total Return (%)",
        "CAGR (%)",
        "Annualized Volatility (%)",
        "Maximum Drawdown (%)",
    ]:
        formatted[col] = formatted[col].map(
            lambda x: f"{x:,.2f}%" if pd.notna(x) else "N/A"
        )

    for col in ["Sharpe Ratio", "Calmar Ratio"]:
        formatted[col] = formatted[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    table_html = formatted.to_html(
        index=False,
        border=0,
        classes="results-table",
        escape=True,
    )

    chart_html = "\n".join([
        to_html(returns_fig, full_html=False, include_plotlyjs=False),
        to_html(risk_fig, full_html=False, include_plotlyjs=False),
        to_html(ratios_fig, full_html=False, include_plotlyjs=False),
    ])

    cagr_text = (
        f"{best['CAGR'] * 100:.2f}%"
        if pd.notna(best["CAGR"]) else "N/A"
    )
    sharpe_text = (
        f"{best['Sharpe Ratio']:.3f}"
        if pd.notna(best["Sharpe Ratio"]) else "N/A"
    )
    drawdown_text = (
        f"{best['Maximum Drawdown'] * 100:.2f}%"
        if pd.notna(best["Maximum Drawdown"]) else "N/A"
    )
    calmar_text = (
        f"{best['Calmar Ratio']:.3f}"
        if pd.notna(best["Calmar Ratio"]) else "N/A"
    )

    observation_text = (
        f"{observations:,}" if observations is not None else "N/A"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Out-of-Sample Strategy Performance</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
:root {{
  --ink: #182230;
  --muted: #667085;
  --line: #e4e7ec;
  --paper: #f5f7fb;
  --card: #ffffff;
  --blue: #2463eb;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: var(--paper);
  color: var(--ink);
}}
header {{
  padding: 34px max(22px, calc((100% - 1200px) / 2));
  background: linear-gradient(120deg, #eef4ff, #ffffff);
  border-bottom: 1px solid var(--line);
}}
.eyebrow {{
  color: var(--blue);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 1.8px;
  text-transform: uppercase;
}}
h1 {{ margin: 10px 0; font-size: clamp(27px, 4vw, 40px); }}
.subtitle {{ color: var(--muted); line-height: 1.6; }}
main {{ max-width: 1200px; margin: 25px auto; padding: 0 20px 40px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}}
.card, .panel {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 15px;
  box-shadow: 0 4px 16px rgba(16, 24, 40, 0.04);
}}
.card {{ padding: 20px; }}
.card-label {{ color: var(--muted); font-size: 13px; }}
.card-value {{ font-size: 25px; font-weight: 750; margin-top: 12px; overflow-wrap: anywhere; }}
.card-note {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
.panel {{ padding: 22px; margin-top: 20px; overflow: hidden; }}
h2 {{ font-size: 20px; margin: 0 0 16px; }}
.results-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
.results-table th {{
  background: #edf3ff; color: #253858; text-align: left;
  padding: 14px 12px; white-space: nowrap;
}}
.results-table td {{ padding: 14px 12px; border-bottom: 1px solid var(--line); }}
.results-table tr:hover td {{ background: #f8faff; }}
.table-wrap {{ overflow-x: auto; }}
.note {{
  margin-top: 18px; padding: 16px; background: #fff8e8;
  border: 1px solid #f2dfb0; border-radius: 10px;
  color: #614b16; line-height: 1.6; font-size: 13px;
}}
footer {{ color: var(--muted); font-size: 12px; text-align: center; padding: 20px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 520px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .card-value {{ font-size: 23px; }}
  .panel {{ padding: 14px; }}
}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Quantitative Finance · Backtesting</div>
  <h1>Out-of-Sample Strategy Performance</h1>
  <div class="subtitle">
    Historical holdout comparison · {test_start} to {test_end}<br>
    {observation_text} daily observations · Interactive charts with hover details
  </div>
</header>
<main>
  <section class="grid">
    <div class="card">
      <div class="card-label">Highest Sharpe Strategy</div>
      <div class="card-value">{best["Strategy"]}</div>
      <div class="card-note">Among the strategies in this report</div>
    </div>
    <div class="card">
      <div class="card-label">CAGR</div>
      <div class="card-value">{cagr_text}</div>
      <div class="card-note">Annualized historical growth</div>
    </div>
    <div class="card">
      <div class="card-label">Sharpe Ratio</div>
      <div class="card-value">{sharpe_text}</div>
      <div class="card-note">Risk-adjusted return, zero risk-free rate</div>
    </div>
    <div class="card">
      <div class="card-label">Maximum Drawdown</div>
      <div class="card-value">{drawdown_text}</div>
      <div class="card-note">Largest peak-to-trough decline</div>
    </div>
  </section>

  <section class="panel">
    <h2>Strategy Metrics Table</h2>
    <div class="table-wrap">{table_html}</div>
    <div class="note">
      <strong>How to read this:</strong> CAGR is annualized growth.
      Volatility measures variation in daily returns. Maximum drawdown is
      negative by convention. Sharpe and Calmar ratios are dimensionless.
      The headline strategy is selected by Sharpe ratio, not by a guarantee
      of future performance.
    </div>
  </section>

  <section class="panel">
    <h2>Interactive Performance Charts</h2>
    {chart_html}
  </section>

  <section class="panel">
    <h2>Methodology and Limitations</h2>
    <p>These results use the supplied out-of-sample summary CSV. The evaluation
    window begins at the holdout date in that file. Earlier prices can be used
    to warm up moving-average and momentum indicators, but returns before the
    holdout date should not be counted in holdout performance.</p>
    <p>Historical backtest performance does not guarantee future results.
    Validate signal timing, missing-price handling, execution assumptions,
    corporate-action treatment, transaction costs, and out-of-sample methodology
    before making investment or resume claims about performance.</p>
  </section>
</main>
<footer>Generated from the local project results · Momentum Strategy Backtesting</footer>
</body>
</html>
"""

    OUTPUT.write_text(html, encoding="utf-8")

    print(f"HTML dashboard saved to: {OUTPUT}")
    print(f"Formatted table saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
