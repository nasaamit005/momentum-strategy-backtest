
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. Paths and configuration
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FULL_RETURNS_FILE = DATA / "daily_returns.csv"
FULL_EQUITY_FILE = DATA / "equity_curves.csv"
OOS_RETURNS_FILE = DATA / "out_of_sample_returns.csv"
OOS_EQUITY_FILE = DATA / "out_of_sample_equity.csv"

OUTPUT_HTML = REPORTS / "risk_analytics_dashboard.html"
OUTPUT_METRICS = REPORTS / "risk_analytics_summary.csv"
OUTPUT_VALIDATION = REPORTS / "risk_data_validation.csv"

TRADING_DAYS = 252
ROLLING_SHORT = 30
ROLLING_LONG = 90

STRATEGIES = ["MA Crossover", "Momentum", "SPY Buy & Hold"]

REPORTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load and validate input files
# ============================================================
def load_series(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column missing from {path.name}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        raise ValueError(f"Invalid dates found in {path.name}")

    if df["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates found in {path.name}")

    missing_columns = set(STRATEGIES) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing strategy columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.sort_values("Date").reset_index(drop=True)

    for col in STRATEGIES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[STRATEGIES].isna().any().any():
        raise ValueError(f"Missing or nonnumeric returns/values in {path.name}")

    if not np.isfinite(df[STRATEGIES].to_numpy(dtype=float)).all():
        raise ValueError(f"Infinite values found in {path.name}")

    return df


full_returns_df = load_series(FULL_RETURNS_FILE)
full_equity_df = load_series(FULL_EQUITY_FILE)
oos_returns_df = load_series(OOS_RETURNS_FILE)
oos_equity_df = load_series(OOS_EQUITY_FILE)

# Verify that each returns series has matching dates in its equity file.
def check_date_alignment(returns_df, equity_df, label):
    if not returns_df["Date"].equals(equity_df["Date"]):
        raise ValueError(
            f"{label}: returns and equity dates do not match. "
            "Resolve alignment before interpreting the dashboard."
        )

check_date_alignment(full_returns_df, full_equity_df, "Full period")
check_date_alignment(oos_returns_df, oos_equity_df, "Out of sample")

# The OOS return and equity series should describe the same daily path.
# We validate their compounded relationship below rather than silently
# replacing the existing equity series.
def compounded_equity(returns):
    return (1.0 + returns).cumprod()

validation_rows = []

for label, returns_df, equity_df in [
    ("Full Period", full_returns_df, full_equity_df),
    ("Out of Sample", oos_returns_df, oos_equity_df),
]:
    for strategy in STRATEGIES:
        returns = returns_df[strategy]
        equity = equity_df[strategy]

        expected = compounded_equity(returns)
        # Equity files may use a different initial-value convention.
        # Compare growth after normalizing the first observation.
        actual_growth = equity / equity.iloc[0]
        expected_growth = expected / expected.iloc[0]

        max_diff = float(
            np.max(np.abs(actual_growth.to_numpy() -
                          expected_growth.to_numpy()))
        )

        validation_rows.append({
            "Period": label,
            "Strategy": strategy,
            "Observations": len(returns),
            "Missing_Returns": int(returns.isna().sum()),
            "Missing_Equity": int(equity.isna().sum()),
            "Max_Normalized_Equity_Difference": max_diff,
            "Growth_Path_Matches": max_diff < 1e-6,
            "Start_Date": str(returns_df["Date"].iloc[0].date()),
            "End_Date": str(returns_df["Date"].iloc[-1].date()),
        })

validation = pd.DataFrame(validation_rows)
validation.to_csv(OUTPUT_VALIDATION, index=False)

# ============================================================
# 3. Risk metric calculations
# ============================================================

def calculate_risk_metrics(returns, equity, dates):
    dates = pd.DatetimeIndex(dates)

    returns = pd.Series(
        np.asarray(returns, dtype=float),
        index=dates
    )

    equity = pd.Series(
        np.asarray(equity, dtype=float),
        index=dates
    )

    # Keep returns and equity aligned on the same dates.
    combined = pd.concat(
        [
            returns.rename("Return"),
            equity.rename("Equity")
        ],
        axis=1
    ).dropna()

    returns = combined["Return"]
    equity = combined["Equity"]

    if len(returns) < 2:
        raise ValueError(
            "Not enough observations to calculate risk metrics."
        )

    if (equity <= 0).any():
        raise ValueError(
            "Equity must remain positive to calculate CAGR."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain infinite values.")

    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError("Equity contains infinite values.")

    # Total return from the actual beginning and ending equity.
    initial_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])

    total_return = final_equity / initial_equity - 1.0

    # Annualize using the number of observed daily returns.
    years = len(returns) / TRADING_DAYS

    cagr = (
    (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    if years > 0
    and equity.iloc[0] > 0
    and equity.iloc[-1] > 0
    else np.nan
)

    # Annualized volatility and Sharpe ratio.
    daily_std = float(returns.std(ddof=1))

    annual_vol = (
        daily_std * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        float(
            returns.mean() / daily_std
            * np.sqrt(TRADING_DAYS)
        )
        if daily_std > 0
        else np.nan
    )

    # Drawdown relative to the running equity peak.
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    # Annualized downside deviation relative to zero daily return.
    negative_returns = np.minimum(
        returns.to_numpy(),
        0.0
    )

    downside_deviation = float(
        np.sqrt(np.mean(negative_returns ** 2))
        * np.sqrt(TRADING_DAYS)
    )

    # Sortino ratio using the zero-return target.
    sortino = (
        float(
            returns.mean() * TRADING_DAYS
            / downside_deviation
        )
        if downside_deviation > 0
        else np.nan
    )

    # Calmar ratio based on corrected CAGR.
    calmar = (
        float(cagr / abs(max_drawdown))
        if pd.notna(cagr) and max_drawdown < 0
        else np.nan
    )

    return {
        "Observations": len(returns),
        "Start Date": str(returns.index.min().date()),
        "End Date": str(returns.index.max().date()),
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Downside Deviation (Annualized)": downside_deviation,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Positive Days (%)": float((returns > 0).mean()),
    }



def make_analysis(returns_df, equity_df, period):
    rows = []
    returns = returns_df.set_index("Date")
    equity = equity_df.set_index("Date")

    for strategy in STRATEGIES:
        metrics = calculate_risk_metrics(
            returns[strategy],
            equity[strategy],
            returns.index
        )
        metrics["Period"] = period
        metrics["Strategy"] = strategy
        rows.append(metrics)

    return pd.DataFrame(rows)


full_metrics = make_analysis(
    full_returns_df, full_equity_df, "Full Period"
)
oos_metrics = make_analysis(
    oos_returns_df, oos_equity_df, "Out of Sample"
)

all_metrics = pd.concat(
    [full_metrics, oos_metrics], ignore_index=True
)

metric_columns = [
    "Period", "Strategy", "Observations", "Start Date", "End Date",
    "Total Return", "CAGR", "Annualized Volatility",
    "Sharpe Ratio", "Sortino Ratio", "Downside Deviation (Annualized)",
    "Maximum Drawdown", "Calmar Ratio", "Positive Days (%)"
]

all_metrics = all_metrics[metric_columns]
all_metrics.to_csv(OUTPUT_METRICS, index=False)

# ============================================================
# 4. Interactive Plotly charts
# ============================================================
def make_equity_frame(returns_df, equity_df, period):
    frames = []
    r = returns_df.set_index("Date")
    e = equity_df.set_index("Date")

    for strategy in STRATEGIES:
        # Normalize each curve to 1 at the start of its displayed period.
        curve = e[strategy] / e[strategy].iloc[0]
        frame = pd.DataFrame({
            "Date": curve.index,
            "Equity": curve.to_numpy(),
            "Strategy": strategy,
            "Period": period,
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


full_curve = make_equity_frame(
    full_returns_df, full_equity_df, "Full Period"
)
oos_curve = make_equity_frame(
    oos_returns_df, oos_equity_df, "Out of Sample"
)

all_curves = pd.concat([full_curve, oos_curve], ignore_index=True)

equity_fig = px.line(
    all_curves,
    x="Date",
    y="Equity",
    color="Strategy",
    facet_row="Period",
    title="Normalized Equity Curves",
    labels={"Equity": "Growth of 1 unit", "Date": ""},
    hover_data={"Equity": ":.3f", "Strategy": True, "Period": True},
    template="plotly_white",
)
equity_fig.update_layout(height=760, hovermode="x unified")
equity_fig.update_yaxes(tickformat=".2f")

# Drawdowns calculated independently within each period.
drawdown_frames = []

for period, returns_df, equity_df in [
    ("Full Period", full_returns_df, full_equity_df),
    ("Out of Sample", oos_returns_df, oos_equity_df),
]:
    e = equity_df.set_index("Date")

    for strategy in STRATEGIES:
        curve = e[strategy]
        dd = curve / curve.cummax() - 1.0
        drawdown_frames.append(pd.DataFrame({
            "Date": dd.index,
            "Drawdown": dd.to_numpy(),
            "Strategy": strategy,
            "Period": period,
        }))

drawdown_data = pd.concat(drawdown_frames, ignore_index=True)

drawdown_fig = px.area(
    drawdown_data,
    x="Date",
    y="Drawdown",
    color="Strategy",
    facet_row="Period",
    title="Portfolio Drawdowns Through Time",
    labels={"Drawdown": "Drawdown from prior peak", "Date": ""},
    hover_data={"Drawdown": ":.2%", "Strategy": True, "Period": True},
    template="plotly_white",
)
drawdown_fig.update_layout(height=760, hovermode="x unified")
drawdown_fig.update_yaxes(tickformat=".0%")

# Rolling metrics use only past and current observations.
rolling_frames = []

for period, returns_df in [
    ("Full Period", full_returns_df),
    ("Out of Sample", oos_returns_df),
]:
    temp = returns_df.set_index("Date")

    for strategy in STRATEGIES:
        r = temp[strategy]

        short_vol = r.rolling(ROLLING_SHORT).std() * np.sqrt(TRADING_DAYS)
        long_vol = r.rolling(ROLLING_LONG).std() * np.sqrt(TRADING_DAYS)

        long_mean = r.rolling(ROLLING_LONG).mean()
        long_std = r.rolling(ROLLING_LONG).std()
        rolling_sharpe = (
            long_mean / long_std * np.sqrt(TRADING_DAYS)
        ).where(long_std > 0)

        negative_sq = r.clip(upper=0) ** 2
        downside_dev = np.sqrt(
            negative_sq.rolling(ROLLING_LONG).mean()
        ) * np.sqrt(TRADING_DAYS)

        rolling_sortino = (
            long_mean * TRADING_DAYS / downside_dev
        ).where(downside_dev > 0)

        for label, series in [
            (f"{ROLLING_SHORT}-Day Volatility", short_vol),
            (f"{ROLLING_LONG}-Day Volatility", long_vol),
            (f"{ROLLING_LONG}-Day Sharpe", rolling_sharpe),
            (f"{ROLLING_LONG}-Day Sortino", rolling_sortino),
        ]:
            rolling_frames.append(pd.DataFrame({
                "Date": series.index,
                "Value": series.to_numpy(),
                "Metric": label,
                "Strategy": strategy,
                "Period": period,
            }))

rolling_data = pd.concat(rolling_frames, ignore_index=True)

vol_data = rolling_data[
    rolling_data["Metric"].isin([
        f"{ROLLING_SHORT}-Day Volatility",
        f"{ROLLING_LONG}-Day Volatility",
    ])
]

vol_fig = px.line(
    vol_data,
    x="Date",
    y="Value",
    color="Strategy",
    facet_row="Period",
    line_dash="Metric",
    title="Rolling Annualized Volatility",
    labels={"Value": "Annualized volatility", "Date": ""},
    hover_data={"Value": ":.2%", "Metric": True, "Period": True},
    template="plotly_white",
)
vol_fig.update_layout(height=760, hovermode="x unified")
vol_fig.update_yaxes(tickformat=".0%")

ratio_data = rolling_data[
    rolling_data["Metric"].isin([
        f"{ROLLING_LONG}-Day Sharpe",
        f"{ROLLING_LONG}-Day Sortino",
    ])
]

ratio_fig = px.line(
    ratio_data,
    x="Date",
    y="Value",
    color="Strategy",
    facet_row="Period",
    line_dash="Metric",
    title="Rolling 90-Day Sharpe and Sortino Ratios",
    labels={"Value": "Rolling ratio", "Date": ""},
    hover_data={"Value": ":.3f", "Metric": True, "Period": True},
    template="plotly_white",
)
ratio_fig.update_layout(height=760, hovermode="x unified")
ratio_fig.add_hline(y=0, line_dash="dot", line_color="gray")

# Compare summary metrics by strategy and period.
risk_compare = px.bar(
    all_metrics,
    x="Strategy",
    y="Maximum Drawdown",
    color="Period",
    barmode="group",
    title="Maximum Drawdown by Strategy and Period",
    labels={"Maximum Drawdown": "Maximum drawdown", "Strategy": ""},
    hover_data={
        "Maximum Drawdown": ":.2%",
        "CAGR": ":.2%",
        "Sharpe Ratio": ":.3f",
        "Sortino Ratio": ":.3f",
        "Period": True,
    },
    template="plotly_white",
)
risk_compare.update_yaxes(tickformat=".0%")

# ============================================================
# 5. KPI cards based on out-of-sample metrics
# ============================================================
def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.3f}"

best_cagr = oos_metrics.loc[oos_metrics["CAGR"].idxmax()]
best_sharpe = oos_metrics.loc[oos_metrics["Sharpe Ratio"].idxmax()]
best_sortino = oos_metrics.loc[oos_metrics["Sortino Ratio"].idxmax()]
least_drawdown = oos_metrics.loc[oos_metrics["Maximum Drawdown"].idxmax()]

def kpi(label, value, subtitle):
    return f"""
    <div class="kpi">
      <div class="label">{html.escape(label)}</div>
      <div class="value">{html.escape(value)}</div>
      <div class="sub">{html.escape(subtitle)}</div>
    </div>
    """

cards = "".join([
    kpi("Highest OOS CAGR", fmt_pct(best_cagr["CAGR"]),
        best_cagr["Strategy"]),
    kpi("Highest OOS Sharpe", fmt_num(best_sharpe["Sharpe Ratio"]),
        best_sharpe["Strategy"]),
    kpi("Highest OOS Sortino", fmt_num(best_sortino["Sortino Ratio"]),
        best_sortino["Strategy"]),
    kpi("Smallest OOS Drawdown", fmt_pct(least_drawdown["Maximum Drawdown"]),
        least_drawdown["Strategy"]),
])

# ============================================================
# 6. Interactive searchable/sortable metrics table
# ============================================================
table_rows = []

for _, row in all_metrics.iterrows():
    table_rows.append({
        "Period": row["Period"],
        "Strategy": row["Strategy"],
        "Observations": int(row["Observations"]),
        "Start Date": row["Start Date"],
        "End Date": row["End Date"],
        "Total Return": fmt_pct(row["Total Return"]),
        "CAGR": fmt_pct(row["CAGR"]),
        "Annualized Volatility": fmt_pct(row["Annualized Volatility"]),
        "Sharpe Ratio": fmt_num(row["Sharpe Ratio"]),
        "Sortino Ratio": fmt_num(row["Sortino Ratio"]),
        "Downside Deviation": fmt_pct(row["Downside Deviation (Annualized)"]),
        "Maximum Drawdown": fmt_pct(row["Maximum Drawdown"]),
        "Calmar Ratio": fmt_num(row["Calmar Ratio"]),
        "Positive Days": fmt_pct(row["Positive Days (%)"]),
    })

table_columns = list(table_rows[0].keys())
table_json = json.dumps(table_rows).replace("</", "<\\/")

# ============================================================
# 7. Save a self-contained interactive HTML report
# ============================================================
chart_html = [
    equity_fig.to_html(full_html=False, include_plotlyjs=True),
    drawdown_fig.to_html(full_html=False, include_plotlyjs=False),
    vol_fig.to_html(full_html=False, include_plotlyjs=False),
    ratio_fig.to_html(full_html=False, include_plotlyjs=False),
    risk_compare.to_html(full_html=False, include_plotlyjs=False),
]

html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk Analytics Dashboard</title>
<style>
:root {{
  --bg: #f4f7fb; --panel: #ffffff; --text: #172b4d;
  --muted: #65758b; --border: #dce4ee; --accent: #2563eb;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
}}
header {{
  padding: 30px 4%; background: white; border-bottom: 1px solid var(--border);
}}
header h1 {{ margin: 0 0 8px; font-size: 29px; }}
header p {{ color: var(--muted); margin: 0; line-height: 1.6; }}
main {{ max-width: 1550px; padding: 24px; margin: auto; }}
h2 {{ font-size: 20px; margin: 28px 0 14px; }}
.kpis {{
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px;
}}
.kpi, .panel, .chart {{
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 14px; box-shadow: 0 3px 12px rgba(30,50,80,.04);
}}
.kpi {{ padding: 20px; }}
.label {{ color: var(--muted); font-size: 13px; }}
.value {{ font-size: 26px; font-weight: 750; margin: 10px 0 6px; }}
.sub {{ color: var(--muted); font-size: 13px; }}
.charts {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
.chart {{ padding: 8px; overflow: hidden; min-width: 0; }}
.panel {{ padding: 20px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }}
input, select, button {{
  padding: 10px 12px; border: 1px solid var(--border);
  border-radius: 8px; font-size: 14px; background: white; color: var(--text);
}}
input {{ flex: 1; min-width: 220px; }}
button {{ background: var(--accent); color: white; cursor: pointer; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ background: #f1f5fa; }}
th button {{ border: 0; background: transparent; color: var(--text); padding: 0; font-weight: 700; }}
tbody tr:hover {{ background: #f5f9ff; }}
.note {{ color: var(--muted); font-size: 13px; line-height: 1.7; }}
footer {{ padding: 24px 4%; color: var(--muted); font-size: 12px; }}
@media(max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
@media(max-width: 520px) {{ main {{ padding: 12px; }} .kpis {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Risk Analytics Dashboard</h1>
  <p>Interactive portfolio risk and performance analysis · Full period and out of sample</p>
</header>
<main>
  <h2>Out-of-Sample Highlights</h2>
  <section class="kpis">{cards}</section>

  <h2>Portfolio Growth</h2>
  <section class="charts"><div class="chart">{chart_html[0]}</div></section>

  <h2>Drawdown Analysis</h2>
  <section class="charts"><div class="chart">{chart_html[1]}</div></section>

  <h2>Rolling Risk Measures</h2>
  <section class="charts">
    <div class="chart">{chart_html[2]}</div>
    <div class="chart">{chart_html[3]}</div>
  </section>

  <h2>Maximum Drawdown Comparison</h2>
  <section class="charts"><div class="chart">{chart_html[4]}</div></section>

  <h2>Detailed Risk Metrics</h2>
  <section class="panel">
    <p class="note">Search, filter, sort columns, and export the currently displayed rows.</p>
    <div class="controls">
      <input id="search" type="search" placeholder="Search strategy or metric...">
      <select id="period">
        <option value="">All periods</option>
        <option value="Full Period">Full Period</option>
        <option value="Out of Sample">Out of Sample</option>
      </select>
      <button id="export">Export displayed CSV</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr id="head"></tr></thead>
        <tbody id="body"></tbody>
      </table>
    </div>
    <p id="count" class="note"></p>
  </section>

  <h2>Data Validation</h2>
  <section class="panel">
    <p class="note">The dashboard also saves risk_data_validation.csv, which checks whether
    normalized equity growth matches compounded daily returns. Review this file before
    treating the reported metrics as validated.</p>
    <p class="note">Rolling 30-day and 90-day volatility are annualized using 252 trading
    days. Rolling Sharpe and Sortino use a zero-return reference and trailing 90-day
    windows. Sortino uses downside deviation calculated from negative daily returns.
    These are research conventions, not universal definitions.</p>
    <p class="note">Full-period and out-of-sample metrics use different date ranges.
    The charts normalize equity to 1 at the start of each displayed period, so they
    compare growth rather than absolute portfolio capital.</p>
  </section>
</main>
<footer>
  Generated from your local CSV files. Source reports and existing charts are preserved.
</footer>
<script>
const rows = {table_json};
const cols = {json.dumps(table_columns)};
let sortKey = "";
let sortAsc = true;

const head = document.getElementById("head");
cols.forEach(col => {{
  const th = document.createElement("th");
  const button = document.createElement("button");
  button.textContent = col + " ↕";
  button.onclick = () => {{
    if (sortKey === col) sortAsc = !sortAsc;
    else {{ sortKey = col; sortAsc = true; }}
    render();
  }};
  th.appendChild(button);
  head.appendChild(th);
}});

function filteredRows() {{
  const query = document.getElementById("search").value.toLowerCase();
  const period = document.getElementById("period").value;

  let data = rows.filter(row =>
    (!period || row.Period === period) &&
    Object.values(row).some(value =>
      String(value).toLowerCase().includes(query)
    )
  );

  if (sortKey) {{
    data.sort((a, b) => {{
      const x = String(a[sortKey] ?? "");
      const y = String(b[sortKey] ?? "");
      const nx = Number(x.replace("%", ""));
      const ny = Number(y.replace("%", ""));
      const numeric = x !== "" && y !== "" &&
        !Number.isNaN(nx) && !Number.isNaN(ny);
      const result = numeric ? nx - ny : x.localeCompare(y);
      return sortAsc ? result : -result;
    }});
  }}
  return data;
}}

function render() {{
  const data = filteredRows();
  const body = document.getElementById("body");
  body.replaceChildren();

  data.forEach(row => {{
    const tr = document.createElement("tr");
    cols.forEach(col => {{
      const td = document.createElement("td");
      td.textContent = row[col] ?? "";
      tr.appendChild(td);
    }});
    body.appendChild(tr);
  }});

  document.getElementById("count").textContent =
    "Showing " + data.length + " of " + rows.length + " rows.";
}}

document.getElementById("search").addEventListener("input", render);
document.getElementById("period").addEventListener("change", render);

document.getElementById("export").addEventListener("click", () => {{
  const data = filteredRows();
  const quote = value => '"' + String(value ?? "").replace(/"/g, '""') + '"';
  const csv = [cols.map(quote).join(","),
    ...data.map(row => cols.map(col => quote(row[col])).join(","))]
    .join("\\r\\n");

  const blob = new Blob([csv], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "risk_analytics_filtered.csv";
  a.click();
  URL.revokeObjectURL(url);
}});

render();
</script>
</body>
</html>
"""

OUTPUT_HTML.write_text(html_page, encoding="utf-8")

print("\nRISK ANALYTICS DASHBOARD CREATED")
print(f"HTML dashboard: {OUTPUT_HTML}")
print(f"Risk metrics:    {OUTPUT_METRICS}")
print(f"Data validation: {OUTPUT_VALIDATION}")
print("\nValidation results:")
print(validation.to_string(index=False))
print("\nRisk metrics:")
print(all_metrics.to_string(index=False))
