# Pro MF Analyzer: SIP vs Lumpsum

An advanced, realistic mutual fund calculator that compares Systematic Investment Plan (SIP) and Lumpsum investment strategies using real Indian AMFI historical data.

This tool goes beyond basic calculators by accounting for real-world financial friction, offering features like XIRR calculation, India LTCG tax simulation, inflation-adjusted returns, and an Optimal SIP Analyzer.

## 🚀 Features

- **Real AMFI Data**: Pulls actual historical NAV data directly from the Association of Mutual Funds in India (AMFI).
- **Core Comparison**: Compares the exact profitability and absolute return of a Lumpsum investment vs. a Monthly SIP over any given timeframe.
- **XIRR (Extended Internal Rate of Return)**: Computes the industry-standard annualized return for SIPs and Lumpsum based on exact cashflow dates.
- **Tax Impact Simulation**: Simulates India's Long Term Capital Gains (LTCG) tax (12.5% above ₹1.25 Lakh exemption) to show your true post-tax take-home profit.
- **Inflation Adjustment**: Discounts the final corpus by an expected inflation rate (e.g., 6%) to show the real purchasing power of the money.
- **Step-Up SIP**: Allows modeling realistic scenarios where your monthly SIP amount increases by a specified percentage every year.
- **Optimal SIP Analyzer**: Calculates exactly how much you need to invest via SIP per month to beat the Lumpsum profit for a specific timeframe.
- **Interactive Dashboard**: A clean, responsive Streamlit dashboard with interactive charts for visualizing portfolio growth over time.
- **CLI Support**: A lightweight command-line script for quick, detailed terminal output.

## 📋 Prerequisites

- Python 3.9+
- `pip` package manager

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pro-mf-analyzer.git
   cd pro-mf-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `streamlit`, `pandas`, `plotly`, `scipy`, and `mftool` installed)*

## 💻 Usage

### Streamlit Dashboard (Recommended)
Run the interactive web dashboard:
```bash
streamlit run sip_vs_lumpsum_dashboard.py
```
*(Or use `py -m streamlit run sip_vs_lumpsum_dashboard.py` on Windows)*

### Command Line Interface
Run the CLI script for a detailed terminal printout:
```bash
python sip_vs_lumsum.py
```
*(Or use `py sip_vs_lumsum.py` on Windows)*

## 📊 How It Works
- The tool applies a generic stamp duty (`0.005%`) on all purchases to mimic real-world friction.
- SIP installments assume purchases on the 1st day of the month or the closest available trading day.
- Rupee Cost Averaging metrics are calculated to show whether the market conditions favored SIPs (lowering the average buy price) or Lumpsums (buying early before a bull run).

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
