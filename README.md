## Key Financial Formulas

### 1. Simple Return

The simple return measures the percentage change in an asset's price between two consecutive periods.

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

### 2. Cumulative Return

Cumulative return measures the total portfolio return over the entire investment period.

$$
R_{\mathrm{cum}} = \frac{V_T}{V_0} - 1
$$

### 3. Annualized Volatility

Annualized volatility estimates the yearly variability of returns using daily return volatility.

$$
\sigma_{\mathrm{annual}} = \sigma_{\mathrm{daily}} \sqrt{252}
$$

### 4. Sharpe Ratio

The Sharpe ratio measures risk-adjusted performance by comparing excess annual return with annualized volatility.

$$
S = \frac{R_{\mathrm{annual}} - R_f}{\sigma_{\mathrm{annual}}}
$$

### 5. Maximum Drawdown

Drawdown measures the decline in portfolio value relative to its previous running peak.

$$
DD_t = \frac{V_t}{\max_{s \leq t} V_s} - 1
$$

Maximum drawdown is the largest peak-to-trough decline, expressed as a negative return or zero.

$$
MDD = \min_t DD_t
$$

### 6. Compound Annual Growth Rate (CAGR)

CAGR estimates the constant annual growth rate that would produce the observed change in portfolio value over the investment period.

$$
\mathrm{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{1}{Y}} - 1
$$

### Definitions of Variables

| Symbol | Name | Description |
|---|---|---|
| $R_t$ | Simple return | Return at time $t$ |
| $P_t$ | Asset price | Asset price at time $t$ |
| $P_{t-1}$ | Previous asset price | Asset price at time $t-1$ |
| $R_{\mathrm{cum}}$ | Cumulative return | Total return over the investment period |
| $V_0$ | Initial portfolio value | Portfolio value at the beginning |
| $V_T$ | Final portfolio value | Portfolio value at the end |
| $\sigma_{\mathrm{daily}}$ | Daily volatility | Standard deviation of daily returns |
| $\sigma_{\mathrm{annual}}$ | Annualized volatility | Annualized standard deviation of returns |
| $R_{\mathrm{annual}}$ | Annualized return | Portfolio return expressed annually |
| $R_f$ | Risk-free rate | Annual risk-free rate |
| $DD_t$ | Drawdown | Decline from the previous running peak |
| $MDD$ | Maximum drawdown | Largest peak-to-trough decline |
| $Y$ | Investment period | Investment duration in years |
| $252$ | Trading days per year | Assumed number of trading days annually |

**Note:** These are standard financial formulas. Verify the exact calculations and assumptions against the project's Python source code and methodology report.