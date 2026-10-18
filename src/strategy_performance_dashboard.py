
from pathlib import Path
import html
import json
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

FULL_FILE = REPORTS / "performance_summary.csv"
OOS_FILE = REPORTS / "out_of_sample_summary.csv"
OUTPUT_FILE = REPORTS / "strategy_performance_dashboard.html"

for file in (FULL_FILE, OOS_FILE):
    if not file.exists():
        raise FileNotFoundError(f"Required report not found: {file}")

# ---------------------------------------------------------
# 2. Load the actual report data
# ---------------------------------------------------------
full = pd.read_csv(FULL_FILE)
oos = pd.read_csv(OOS_FILE)

required = {
    "Strategy", "Total Return", "CAGR",
    "Annualized Volatility", "Sharpe Ratio",
    "Maximum Drawdown", "Calmar Ratio"
}

for name, df in [("Full-period", full), ("Out-of-sample", oos)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} report is missing columns: {sorted(missing)}")

# Ensure numeric columns are numeric.
metrics = [
    "Total Return", "CAGR", "Annualized Volatility",
    "Sharpe Ratio", "Maximum Drawdown", "Calmar Ratio"
]

for df in (full, oos):
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="raise")

full["Evaluation Period"] = "Full Period"
oos["Evaluation Period"] = "Out of Sample"

combined = pd.concat([full, oos], ignore_index=True)

# Add dates to the out-of-sample labels when available.
if {"Test Start", "Test End"}.issubset(oos.columns):
    oos["Period Detail"] = (
        oos["Test Start"].astype(str) + " to " +
        oos["Test End"].astype(str)
    )
else:
    oos["Period Detail"] = "Out-of-sample"

full["Period Detail"] = "Full-period report"
combined = pd.concat([full, oos], ignore_index=True)

# ---------------------------------------------------------
# 3. Format values for the interactive table
# Values in return, CAGR, volatility and drawdown columns
# are decimal ratios in the CSV files.
# ---------------------------------------------------------
def pct(value):
    return f"{value:.2%}"

def number(value):
    return f"{value:.3f}"

table_rows = []

for _, row in combined.iterrows():
    table_rows.append({
        "Strategy": str(row["Strategy"]),
        "Period": str(row["Evaluation Period"]),
        "Total Return": pct(row["Total Return"]),
        "CAGR": pct(row["CAGR"]),
        "Annualized Volatility": pct(row["Annualized Volatility"]),
        "Sharpe Ratio": number(row["Sharpe Ratio"]),
        "Maximum Drawdown": pct(row["Maximum Drawdown"]),
        "Calmar Ratio": number(row["Calmar Ratio"]),
    })

columns = list(table_rows[0].keys()) if table_rows else []
table_json = json.dumps(table_rows).replace("</", "<\\/")

# ---------------------------------------------------------
# 4. Build Plotly charts
# ---------------------------------------------------------
period_order = ["Full Period", "Out of Sample"]

cagr_fig = px.bar(
    combined,
    x="Strategy",
    y="CAGR",
    color="Evaluation Period",
    barmode="group",
    category_orders={"Evaluation Period": period_order},
    title="CAGR Comparison: Full Period vs Out of Sample",
    labels={"CAGR": "Compound Annual Growth Rate", "Strategy": ""},
    hover_data={
        "CAGR": ":.2%",
        "Annualized Volatility": ":.2%",
        "Sharpe Ratio": ":.3f",
        "Evaluation Period": True,
    },
    template="plotly_white",
)
cagr_fig.update_yaxes(tickformat=".0%")
cagr_fig.update_layout(legend_title_text="Evaluation Period")

risk_fig = px.scatter(
    combined,
    x="Annualized Volatility",
    y="CAGR",
    color="Strategy",
    symbol="Evaluation Period",
    size=[12] * len(combined),
    hover_name="Strategy",
    hover_data={
        "Evaluation Period": True,
        "CAGR": ":.2%",
        "Annualized Volatility": ":.2%",
        "Sharpe Ratio": ":.3f",
        "Maximum Drawdown": ":.2%",
        "Calmar Ratio": ":.3f",
    },
    title="Risk–Return Map",
    labels={
        "Annualized Volatility": "Annualized Volatility",
        "CAGR": "CAGR",
    },
    template="plotly_white",
)
risk_fig.update_xaxes(tickformat=".0%")
risk_fig.update_yaxes(tickformat=".0%")

sharpe_fig = px.bar(
    combined,
    x="Strategy",
    y="Sharpe Ratio",
    color="Evaluation Period",
    barmode="group",
    category_orders={"Evaluation Period": period_order},
    title="Risk-Adjusted Performance: Sharpe Ratio",
    labels={"Sharpe Ratio": "Sharpe Ratio", "Strategy": ""},
    hover_data={
        "Sharpe Ratio": ":.3f",
        "CAGR": ":.2%",
        "Maximum Drawdown": ":.2%",
        "Evaluation Period": True,
    },
    template="plotly_white",
)
sharpe_fig.update_layout(legend_title_text="Evaluation Period")

drawdown_fig = px.bar(
    combined,
    x="Strategy",
    y="Maximum Drawdown",
    color="Evaluation Period",
    barmode="group",
    category_orders={"Evaluation Period": period_order},
    title="Maximum Drawdown Comparison",
    labels={"Maximum Drawdown": "Maximum Drawdown", "Strategy": ""},
    hover_data={
        "Maximum Drawdown": ":.2%",
        "CAGR": ":.2%",
        "Sharpe Ratio": ":.3f",
        "Evaluation Period": True,
    },
    template="plotly_white",
)
drawdown_fig.update_yaxes(tickformat=".0%")
drawdown_fig.update_layout(legend_title_text="Evaluation Period")

# Include Plotly JS once; subsequent charts reuse it.
charts_html = [
    cagr_fig.to_html(full_html=False, include_plotlyjs=True),
    risk_fig.to_html(full_html=False, include_plotlyjs=False),
    sharpe_fig.to_html(full_html=False, include_plotlyjs=False),
    drawdown_fig.to_html(full_html=False, include_plotlyjs=False),
]

# ---------------------------------------------------------
# 5. KPI cards from out-of-sample results
# ---------------------------------------------------------
oos_best_cagr = oos.loc[oos["CAGR"].idxmax()]
oos_best_sharpe = oos.loc[oos["Sharpe Ratio"].idxmax()]
oos_lowest_drawdown = oos.loc[oos["Maximum Drawdown"].idxmax()]

def card(label, value, subtitle):
    return f"""
    <div class="kpi">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(value)}</div>
      <div class="kpi-sub">{html.escape(subtitle)}</div>
    </div>
    """

cards = "".join([
    card("Highest OOS CAGR", pct(oos_best_cagr["CAGR"]),
         oos_best_cagr["Strategy"]),
    card("Highest OOS Sharpe", number(oos_best_sharpe["Sharpe Ratio"]),
         oos_best_sharpe["Strategy"]),
    card("Smallest OOS Drawdown", pct(oos_lowest_drawdown["Maximum Drawdown"]),
         oos_lowest_drawdown["Strategy"]),
    card("Strategies Compared", str(combined["Strategy"].nunique()),
         "Across two evaluation periods"),
])

# ---------------------------------------------------------
# 6. Complete standalone HTML dashboard
# ---------------------------------------------------------
page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Performance Dashboard</title>
<style>
:root {{
  --bg: #f4f7fb;
  --panel: #ffffff;
  --text: #172b4d;
  --muted: #66758a;
  --border: #dce4ee;
  --accent: #2563eb;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
}}
header {{
  background: #ffffff;
  border-bottom: 1px solid var(--border);
  padding: 28px 4%;
}}
header h1 {{ margin: 0 0 8px; font-size: 28px; }}
header p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
main {{ max-width: 1500px; margin: auto; padding: 24px; }}
.section-title {{ margin: 26px 0 14px; font-size: 20px; }}
.kpis {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}}
.kpi, .panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 3px 12px rgba(30, 50, 80, 0.04);
}}
.kpi {{ padding: 20px; }}
.kpi-label {{ color: var(--muted); font-size: 13px; }}
.kpi-value {{ font-size: 27px; font-weight: 750; margin: 10px 0 5px; }}
.kpi-sub {{ color: var(--muted); font-size: 13px; }}
.chart-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}}
.chart {{
  background: white;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 8px;
  min-width: 0;
  overflow: hidden;
}}
.panel {{ padding: 20px; }}
.controls {{
  display: flex; flex-wrap: wrap; gap: 10px;
  margin: 12px 0 16px;
}}
input, select, button {{
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: white;
  color: var(--text);
  font-size: 14px;
}}
input {{ flex: 1; min-width: 220px; }}
button {{
  cursor: pointer; background: var(--accent);
  color: white; border-color: var(--accent);
}}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{
  padding: 12px 10px; border-bottom: 1px solid var(--border);
  text-align: left; white-space: nowrap;
}}
th {{ background: #f1f5fa; position: sticky; top: 0; }}
th button {{
  background: transparent; color: var(--text);
  border: 0; padding: 0; font-weight: 700;
}}
tbody tr:hover {{ background: #f5f9ff; }}
.note {{
  margin-top: 18px; color: var(--muted);
  font-size: 13px; line-height: 1.7;
}}
footer {{ padding: 25px 4%; color: var(--muted); font-size: 12px; }}
@media(max-width: 900px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .chart-grid {{ grid-template-columns: 1fr; }}
}}
@media(max-width: 520px) {{
  main {{ padding: 14px; }}
  .kpis {{ grid-template-columns: 1fr; }}
  header h1 {{ font-size: 23px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Strategy Performance Analytics</h1>
  <p>Interactive backtest report · Full-period and out-of-sample evaluation ·
     Source: your saved strategy summary CSV files</p>
</header>
<main>
  <h2 class="section-title">Out-of-Sample Highlights</h2>
  <section class="kpis">{cards}</section>

  <h2 class="section-title">Interactive Performance Charts</h2>
  <section class="chart-grid">
    <div class="chart">{charts_html[0]}</div>
    <div class="chart">{charts_html[1]}</div>
    <div class="chart">{charts_html[2]}</div>
    <div class="chart">{charts_html[3]}</div>
  </section>

  <h2 class="section-title">Detailed Strategy Metrics</h2>
  <section class="panel">
    <p class="note">Search strategies, filter by evaluation period, click a column
    heading to sort, or export the displayed rows as CSV.</p>
    <div class="controls">
      <input id="search" type="search" placeholder="Search strategy or metric...">
      <select id="period">
        <option value="">All evaluation periods</option>
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

  <h2 class="section-title">How to Interpret This Dashboard</h2>
  <section class="panel note">
    <p><b>CAGR:</b> annualized compounded growth rate over the evaluation period.</p>
    <p><b>Sharpe ratio:</b> average excess-return measure relative to volatility;
    the source report uses its existing calculation assumptions.</p>
    <p><b>Maximum drawdown:</b> largest peak-to-trough decline in the tested
    portfolio equity curve.</p>
    <p><b>Annualized volatility:</b> annualized variability of returns.</p>
    <p><b>Important:</b> these charts compare summary statistics, not daily equity
    curves. Full-period and out-of-sample dates differ, and historical backtest
    results do not establish future profitability. Transaction costs, taxes,
    slippage, survivorship bias, and other implementation assumptions can affect
    real-world outcomes.</p>
  </section>
</main>
<footer>
  Generated locally from performance_summary.csv and out_of_sample_summary.csv.
  No changes were made to the source reports.
</footer>
<script>
const rows = {table_json};
const cols = {json.dumps(columns)};
let sortKey = "";
let sortAsc = true;

const head = document.getElementById("head");
cols.forEach(col => {{
  const th = document.createElement("th");
  const btn = document.createElement("button");
  btn.textContent = col + " ↕";
  btn.onclick = () => {{
    if (sortKey === col) sortAsc = !sortAsc;
    else {{ sortKey = col; sortAsc = true; }}
    render();
  }};
  th.appendChild(btn);
  head.appendChild(th);
}});

function getFiltered() {{
  const q = document.getElementById("search").value.toLowerCase();
  const p = document.getElementById("period").value;
  let data = rows.filter(r =>
    (!p || r["Period"] === p) &&
    Object.values(r).some(v => String(v).toLowerCase().includes(q))
  );
  if (sortKey) {{
    data.sort((a, b) => {{
      const x = a[sortKey], y = b[sortKey];
      const nx = Number(String(x).replace("%", ""));
      const ny = Number(String(y).replace("%", ""));
      const numeric = x !== "" && y !== "" &&
        !Number.isNaN(nx) && !Number.isNaN(ny);
      const cmp = numeric ? nx - ny : String(x).localeCompare(String(y));
      return sortAsc ? cmp : -cmp;
    }});
  }}
  return data;
}}

function render() {{
  const data = getFiltered();
  const tbody = document.getElementById("body");
  tbody.replaceChildren();
  data.forEach(row => {{
    const tr = document.createElement("tr");
    cols.forEach(col => {{
      const td = document.createElement("td");
      td.textContent = row[col] ?? "";
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  document.getElementById("count").textContent =
    "Showing " + data.length + " of " + rows.length + " rows.";
}}

document.getElementById("search").addEventListener("input", render);
document.getElementById("period").addEventListener("change", render);
document.getElementById("export").addEventListener("click", () => {{
  const data = getFiltered();
  const quote = v => '"' + String(v ?? "").replace(/"/g, '""') + '"';
  const csv = [cols.map(quote).join(","),
    ...data.map(r => cols.map(c => quote(r[c])).join(","))].join("\\r\\n");
  const blob = new Blob([csv], {{type: "text/csv;charset=utf-8;"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "strategy_performance_filtered.csv";
  a.click();
  URL.revokeObjectURL(url);
}});
render();
</script>
</body>
</html>
"""

OUTPUT_FILE.write_text(page, encoding="utf-8")
print("\nDashboard created successfully!")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Strategies: {combined['Strategy'].nunique()}")
print(f"Rows displayed in dashboard: {len(combined)}")
