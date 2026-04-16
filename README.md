# Pro MF Analyzer: SIP vs Lumpsum

An advanced, realistic mutual fund calculator that compares Systematic Investment Plan (SIP) and Lumpsum investment strategies using real Indian AMFI historical data.

This tool goes beyond basic calculators by accounting for real-world financial friction, offering features like XIRR calculation, India LTCG tax simulation, inflation-adjusted returns, rolling returns consistency analysis, drawdown risk visualization, portfolio correlation warnings, and a goal-based reverse calculator.

## 🚀 Features

### Phase 1 — Core Analytics
- **Real AMFI Data**: Pulls actual historical NAV data directly from the Association of Mutual Funds in India (AMFI).
- **Core Comparison**: Compares the exact profitability and absolute return of a Lumpsum investment vs. a Monthly SIP over any given timeframe.
- **XIRR (Extended Internal Rate of Return)**: Computes the industry-standard annualized return for SIPs and Lumpsum based on exact cashflow dates.
- **Tax Impact Simulation**: Simulates India's Long Term Capital Gains (LTCG) tax (12.5% above ₹1.25 Lakh exemption) to show your true post-tax take-home profit.
- **Inflation Adjustment**: Discounts the final corpus by an expected inflation rate (e.g., 6%) to show the real purchasing power of the money.
- **Step-Up SIP**: Allows modeling realistic scenarios where your monthly SIP amount increases by a specified percentage every year.
- **Optimal SIP Analyzer**: Calculates exactly how much you need to invest via SIP per month to beat the Lumpsum profit for a specific timeframe.

### Phase 2 — Advanced Analytics (Pro Tier)
- **Rolling Returns Analysis**: Calculates CAGR for every possible 3/5/7/10-year window in the fund's history. Shows average, best, worst, and % positive periods — revealing a fund's true consistency beyond point-to-point luck.
- **Drawdown Visualizer**: Tracks peak-to-trough drops with an interactive chart showing Max Drawdown %, peak/bottom dates, and recovery time in days — so you understand the actual "stomach-churn" of a fund.
- **Portfolio Overlap / Correlation**: When you select 2+ funds, computes a Pearson correlation matrix on daily returns and warns you if any pair has >85% correlation — preventing illusion of diversification.
- **Goal-Based Reverse Calculator**: Enter a financial goal (e.g., "₹1 Crore in 10 years") and the app reverse-computes the required monthly SIP and one-time lumpsum based on the fund's historical CAGR, with a sensitivity table across 5–25 year horizons.

### Platform Support
- **Interactive Dashboard**: A clean, responsive Streamlit dashboard with interactive Plotly charts for visualizing portfolio growth, rolling returns, drawdowns, and correlation heatmaps.
- **CLI Support**: A lightweight command-line script for quick, detailed terminal output with all analytics.

## 📋 Prerequisites

- Python 3.9+
- `pip` package manager

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TREEofbusybeaver/pro-mf-analyzer.git
   cd pro-mf-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy plotly scipy mftool
   ```

## 💻 Usage

### Streamlit Dashboard (Recommended)
Run the interactive web dashboard:
```bash
streamlit run sip_vs_lumpsum_dashboard.py
```

### Command Line Interface
Run the CLI script for a detailed terminal printout:
```bash
python sip_vs_lumpsum.py
```

## 📊 How It Works
- The tool applies a generic stamp duty (`0.005%`) on all purchases to mimic real-world friction.
- SIP installments assume purchases on the 1st day of the month or the closest available trading day.
- Rupee Cost Averaging metrics are calculated to show whether the market conditions favored SIPs (lowering the average buy price) or Lumpsums (buying early before a bull run).
- Rolling returns use full historical data (not just the selected timeframe) for a comprehensive consistency picture.
- Correlation analysis uses daily returns alignment to ensure accurate overlap measurement.

## 🗓️ Changelog

### v2.0.0 — Phase 2: Advanced Analytics (2026-04-16)
- Added Rolling Returns Analysis (3/5/7/10-year configurable window)
- Added Drawdown Visualizer with max drawdown, peak/bottom dates, recovery tracking
- Added Portfolio Overlap / Correlation heatmap with >85% overlap warnings
- Added Goal-Based Reverse Calculator with sensitivity table
- New sidebar controls for rolling window and goal inputs

### v1.0.0 — Phase 1: Core Calculator (2026-04-15)
- SIP vs Lumpsum comparison with real AMFI data
- XIRR calculation, LTCG tax simulation, inflation adjustment
- Step-Up SIP, Optimal SIP Analyzer
- Streamlit dashboard + CLI tool

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
