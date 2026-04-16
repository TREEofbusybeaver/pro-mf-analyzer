import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mftool import Mftool
import datetime
from scipy.optimize import brentq

# --- Page Configuration ---
st.set_page_config(page_title="Pro MF Analyzer", page_icon="📈", layout="wide")
st.title("📈 Advanced Mutual Fund Analyzer")
st.markdown("Compare SIP vs Lumpsum strategies with real AMFI historical data.")

# --- Helper Functions ---

def calculate_xirr(cashflows, dates):
    """Calculate XIRR using scipy's brentq root-finder."""
    if not cashflows or len(cashflows) != len(dates):
        return None
    day_counts = [(d - dates[0]).days for d in dates]
    def npv(rate):
        return sum(cf / (1 + rate) ** (dc / 365.0) for cf, dc in zip(cashflows, day_counts))
    try:
        rate = brentq(npv, -0.99, 10.0, maxiter=1000)
        return rate * 100
    except (ValueError, RuntimeError):
        return None

def calculate_ltcg_tax(profit, exemption=125000, rate=0.125):
    """India LTCG: 12.5% on gains above ₹1.25 Lakh."""
    if profit <= 0:
        return 0, profit
    taxable = max(0, profit - exemption)
    tax = taxable * rate
    return tax, profit - tax

def inflation_adjusted_value(nominal_value, inflation_rate, years):
    """Discount nominal value by inflation to get real purchasing power."""
    if years <= 0 or inflation_rate <= 0:
        return nominal_value
    return nominal_value / ((1 + inflation_rate / 100) ** years)


def calculate_rolling_returns(df, window_years=3):
    """Calculate rolling CAGR for every possible window_years-length period."""
    window_days = int(window_years * 365.25)
    results = []
    nav_series = df['nav']
    dates = df.index
    for i in range(len(dates)):
        start_date = dates[i]
        end_date = start_date + pd.Timedelta(days=window_days)
        future_dates = dates[dates >= end_date]
        if len(future_dates) == 0:
            break
        actual_end = future_dates[0]
        start_nav = nav_series.iloc[i]
        end_nav = nav_series.loc[actual_end]
        actual_years = (actual_end - start_date).days / 365.25
        if actual_years > 0 and start_nav > 0:
            cagr = ((end_nav / start_nav) ** (1 / actual_years) - 1) * 100
            results.append({'date': start_date, 'cagr': cagr})
    if not results:
        return pd.DataFrame(columns=['date', 'cagr']), {}
    rolling_df = pd.DataFrame(results)
    summary = {
        'avg': rolling_df['cagr'].mean(),
        'median': rolling_df['cagr'].median(),
        'min': rolling_df['cagr'].min(),
        'max': rolling_df['cagr'].max(),
        'min_date': rolling_df.loc[rolling_df['cagr'].idxmin(), 'date'],
        'max_date': rolling_df.loc[rolling_df['cagr'].idxmax(), 'date'],
        'positive_pct': (rolling_df['cagr'] > 0).mean() * 100,
        'window_years': window_years,
        'total_periods': len(rolling_df)
    }
    return rolling_df, summary


def calculate_drawdown(df):
    """Calculate drawdown series from a NAV DataFrame."""
    nav = df['nav']
    running_max = nav.cummax()
    drawdown_pct = ((nav - running_max) / running_max) * 100
    dd_df = pd.DataFrame({
        'nav': nav, 'peak_nav': running_max, 'drawdown_pct': drawdown_pct
    }, index=df.index)
    max_dd_idx = drawdown_pct.idxmin()
    max_dd_value = drawdown_pct.min()
    peak_date = nav[:max_dd_idx].idxmax() if max_dd_idx is not None else None
    peak_nav_at_dd = running_max.loc[max_dd_idx] if max_dd_idx is not None else None
    recovery_date = None
    if max_dd_idx is not None and peak_nav_at_dd is not None:
        post_dd = nav[max_dd_idx:]
        recovered = post_dd[post_dd >= peak_nav_at_dd]
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
    max_dd_info = {
        'max_drawdown_pct': max_dd_value,
        'drawdown_date': max_dd_idx,
        'peak_date': peak_date,
        'recovery_date': recovery_date,
        'recovery_days': (recovery_date - max_dd_idx).days if recovery_date else None
    }
    return dd_df, max_dd_info


def calculate_correlation_matrix(dfs_dict):
    """Calculate Pearson correlation on daily returns between multiple funds."""
    returns_dict = {}
    for name, df in dfs_dict.items():
        daily_returns = df['nav'].pct_change().dropna()
        daily_returns.name = name
        returns_dict[name] = daily_returns
    if len(returns_dict) < 2:
        return pd.DataFrame(), []
    returns_df = pd.DataFrame(returns_dict).dropna()
    if returns_df.empty:
        return pd.DataFrame(), []
    corr_matrix = returns_df.corr()
    warnings = []
    fund_names = list(dfs_dict.keys())
    for i in range(len(fund_names)):
        for j in range(i + 1, len(fund_names)):
            corr_val = corr_matrix.loc[fund_names[i], fund_names[j]]
            if corr_val > 0.85:
                warnings.append(
                    f"⚠️ **HIGH OVERLAP:** *{fund_names[i]}* and *{fund_names[j]}* "
                    f"have **{corr_val:.1%}** correlation — they move almost identically. "
                    f"Adding both provides limited diversification benefit."
                )
    return corr_matrix, warnings


def goal_reverse_calculator(target_amount, years, cagr_pct):
    """Reverse-calculate the required SIP or Lumpsum to reach a financial goal."""
    monthly_rate = (1 + cagr_pct / 100) ** (1/12) - 1
    total_months = int(years * 12)
    required_lumpsum = target_amount / ((1 + cagr_pct / 100) ** years)
    if monthly_rate > 0:
        required_sip = target_amount / (((1 + monthly_rate) ** total_months - 1) / monthly_rate * (1 + monthly_rate))
    else:
        required_sip = target_amount / total_months
    return {
        'required_lumpsum': required_lumpsum,
        'required_monthly_sip': required_sip,
        'target': target_amount,
        'years': years,
        'cagr_used': cagr_pct
    }

# --- Initialize Global Tools & Caching ---
@st.cache_resource
def get_mftool():
    return Mftool()

mf = get_mftool()

@st.cache_data(show_spinner=False)
def get_all_schemes():
    """Fetches the master dictionary to fix the 'Unknown Fund Name' issue."""
    return mf.get_scheme_codes()

@st.cache_data(show_spinner=False)
def fetch_mf_data(scheme_code):
    """Fetches historical data for a specific scheme."""
    try:
        data = mf.get_scheme_historical_nav(scheme_code.strip(), as_json=False)
        if data and 'data' in data:
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.sort_values('date').set_index('date')
            return df
    except Exception as e:
        return None
    return None

# Load the dictionary immediately 
all_schemes_dict = get_all_schemes()

# --- Format the Dictionary for the Search UI ---
searchable_funds = [f"{code} - {name}" for code, name in all_schemes_dict.items()]

# --- Sidebar: User Inputs ---
with st.sidebar:
    st.header("⚙️ Strategy Inputs")
    total_lumpsum = st.number_input("Total Lumpsum Investment (₹)", value=200000, step=10000)
    monthly_sip = st.number_input("Monthly SIP Installment (₹)", value=6000, step=500)
    
    st.subheader("Step-Up SIP")
    annual_step_up_pct = st.slider("Annual Step-Up %", min_value=0, max_value=50, value=10, step=5,
                                    help="Increase your SIP by this % every year (e.g., 10% means Year 2 SIP = Year 1 × 1.10)")
    
    st.subheader("Timeframe")
    start_date = st.date_input("Start Date", datetime.date(2022, 6, 8))
    end_date = st.date_input("End Date", datetime.date(2024, 10, 8))
    
    st.subheader("Advanced Settings")
    inflation_rate = st.slider("Expected Inflation Rate (%)", min_value=0.0, max_value=15.0, value=6.0, step=0.5,
                                help="Used to calculate the real purchasing power of your final corpus")
    
    st.subheader("Optimal SIP Analyzer")
    st.markdown("How much *extra* capital are you willing to invest to beat the Lumpsum profit?")
    multiplier_input = st.slider("Max Budget Limit", min_value=1.0, max_value=3.0, value=1.10, step=0.1, help="1.10 means Max SIP Budget is Lumpsum + 10%")
    
    st.divider()

    # --- Phase 2: Rolling Returns Window ---
    st.subheader("📊 Rolling Returns")
    rolling_window = st.selectbox("Rolling Return Window", options=[3, 5, 7, 10], index=0,
                                   format_func=lambda x: f"{x}-Year Rolling", help="Window size for rolling CAGR analysis")
    
    # --- Phase 2: Goal Calculator ---
    st.divider()
    st.subheader("🎯 Goal Calculator")
    goal_target = st.number_input("Target Amount (₹)", value=10000000, step=1000000, 
                                   help="Enter your financial goal, e.g. ₹1,00,00,000 for 1 Crore")
    goal_years = st.number_input("Time Horizon (Years)", value=10, min_value=1, max_value=40, step=1)

    st.divider()

    st.header("🔍 Search & Add Funds")
    st.markdown("*Type a keyword (e.g., 'Flexi Cap', 'Quant', 'Silver') and select from the dropdown.*")
    
    selected_funds = st.multiselect(
        "Search Mutual Funds", 
        options=searchable_funds,
        placeholder="Type here to search..."
    )
    
    analyze_button = st.button("Run Full Analysis 🚀", type="primary", use_container_width=True)

# --- Main Application Logic ---
if analyze_button:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    scheme_codes = [selection.split(" - ")[0] for selection in selected_funds]
    
    if start_dt >= end_dt:
        st.error("Start date must be before the end date.")
    elif not scheme_codes:
        st.warning("Please search and select at least one mutual fund from the sidebar.")
    else:
        with st.spinner("Crunching historical data and running analyzer..."):
            master_graph_data = pd.DataFrame()
            correlation_dfs = {}  # Phase 2: collect DataFrames for correlation

            for code in scheme_codes:
                df = fetch_mf_data(code)
                fund_name = all_schemes_dict.get(code, f"Unknown Scheme ({code})")
                
                if df is not None:
                    mask = (df.index >= start_dt) & (df.index <= end_dt)
                    df_filtered = df.loc[mask]
                    
                    if not df_filtered.empty:
                        # Store for correlation analysis
                        correlation_dfs[fund_name] = df_filtered.copy()

                        st.subheader(f"📊 {fund_name}")
                        
                        start_nav = df_filtered.iloc[0]['nav']
                        end_nav = df_filtered.iloc[-1]['nav']
                        actual_start = df_filtered.index[0]
                        actual_end = df_filtered.index[-1]
                        investment_years = (actual_end - actual_start).days / 365.25
                        
                        # --- 1. LUMPSUM MATH ---
                        ls_invested_after_tax = total_lumpsum - (total_lumpsum * 0.00005)
                        ls_units = round(ls_invested_after_tax / start_nav, 3) 
                        ls_final_value = ls_units * end_nav
                        ls_profit = ls_final_value - total_lumpsum
                        ls_return_pct = (ls_profit / total_lumpsum) * 100 if total_lumpsum > 0 else 0

                        # Lumpsum XIRR
                        ls_cashflows = [-total_lumpsum, ls_final_value]
                        ls_dates_xirr = [actual_start.to_pydatetime(), actual_end.to_pydatetime()]
                        ls_xirr = calculate_xirr(ls_cashflows, ls_dates_xirr)

                        # Lumpsum Tax
                        ls_tax, ls_post_tax_profit = calculate_ltcg_tax(ls_profit)
                        ls_post_tax_value = total_lumpsum + ls_post_tax_profit

                        # Lumpsum Inflation-Adjusted
                        ls_real_value = inflation_adjusted_value(ls_final_value, inflation_rate, investment_years)

                        # --- 2. SIP MATH (with Step-Up) ---
                        monthly_df = df_filtered.resample('MS').first().dropna()
                        months = len(monthly_df)
                        
                        if months > 0:
                            sip_total_invested = 0
                            sip_total_units = 0
                            sip_cashflows = []
                            sip_dates_xirr = []
                            
                            current_sip = monthly_sip
                            sip_start_date_dt = monthly_df.index[0]
                            
                            units_per_month = []
                            
                            for i, (idx, row) in enumerate(monthly_df.iterrows()):
                                # Step-Up: increase SIP at the start of each new year
                                if annual_step_up_pct > 0 and i > 0:
                                    months_elapsed = (idx.year - sip_start_date_dt.year) * 12 + (idx.month - sip_start_date_dt.month)
                                    year_number = months_elapsed // 12
                                    current_sip = monthly_sip * ((1 + annual_step_up_pct / 100) ** year_number)
                                
                                installment_after_stamp = current_sip - (current_sip * 0.00005)
                                sip_total_invested += current_sip
                                
                                if pd.notna(row['nav']):
                                    units = round(installment_after_stamp / row['nav'], 3)
                                    sip_total_units += units
                                    units_per_month.append(units)
                                    sip_cashflows.append(-current_sip)
                                    sip_dates_xirr.append(idx.to_pydatetime())
                                else:
                                    units_per_month.append(0)
                            
                            sip_final_value = sip_total_units * end_nav
                            sip_profit = sip_final_value - sip_total_invested
                            sip_return_pct = (sip_profit / sip_total_invested) * 100 if sip_total_invested > 0 else 0
                            sip_avg_nav = sip_total_invested / sip_total_units if sip_total_units > 0 else 0

                            # SIP XIRR
                            sip_cashflows.append(sip_final_value)
                            sip_dates_xirr.append(actual_end.to_pydatetime())
                            sip_xirr = calculate_xirr(sip_cashflows, sip_dates_xirr)

                            # SIP Tax
                            sip_tax, sip_post_tax_profit = calculate_ltcg_tax(sip_profit)
                            sip_post_tax_value = sip_total_invested + sip_post_tax_profit

                            # SIP Inflation-Adjusted
                            sip_real_value = inflation_adjusted_value(sip_final_value, inflation_rate, investment_years)

                            # ===== PHASE 1 METRICS =====

                            # --- Metrics Row 1: Core ---
                            st.markdown("##### Core Metrics")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Lumpsum Profit", f"₹{ls_profit:,.0f}", f"{ls_return_pct:.1f}%")
                            col2.metric("SIP Profit", f"₹{sip_profit:,.0f}", f"{sip_return_pct:.1f}%")
                            
                            if sip_profit > ls_profit:
                                col3.metric("Winner (Absolute)", "SIP", f"+₹{(sip_profit - ls_profit):,.0f}")
                            else:
                                col3.metric("Winner (Absolute)", "Lumpsum", f"+₹{(ls_profit - sip_profit):,.0f}")

                            # --- Metrics Row 2: XIRR ---
                            st.markdown("##### XIRR (Annualized Return)")
                            col4, col5, col6 = st.columns(3)
                            col4.metric("Lumpsum XIRR", f"{ls_xirr:.2f}%" if ls_xirr else "N/A")
                            col5.metric("SIP XIRR", f"{sip_xirr:.2f}%" if sip_xirr else "N/A")
                            if ls_xirr and sip_xirr:
                                xirr_winner = "SIP" if sip_xirr > ls_xirr else "Lumpsum"
                                xirr_diff = abs(sip_xirr - ls_xirr)
                                col6.metric("Winner (XIRR)", xirr_winner, f"+{xirr_diff:.2f}%")

                            # --- Metrics Row 3: Post-Tax ---
                            st.markdown("##### Tax Impact (LTCG @ 12.5% above ₹1.25L)")
                            col7, col8, col9 = st.columns(3)
                            col7.metric("Lumpsum Post-Tax", f"₹{ls_post_tax_profit:,.0f}", f"-₹{ls_tax:,.0f} tax")
                            col8.metric("SIP Post-Tax", f"₹{sip_post_tax_profit:,.0f}", f"-₹{sip_tax:,.0f} tax")
                            if sip_post_tax_profit > ls_post_tax_profit:
                                col9.metric("Winner (Post-Tax)", "SIP", f"+₹{(sip_post_tax_profit - ls_post_tax_profit):,.0f}")
                            else:
                                col9.metric("Winner (Post-Tax)", "Lumpsum", f"+₹{(ls_post_tax_profit - sip_post_tax_profit):,.0f}")

                            # --- Metrics Row 4: Inflation Adjusted ---
                            st.markdown(f"##### Inflation-Adjusted Value ({inflation_rate}% p.a.)")
                            col10, col11, col12 = st.columns(3)
                            col10.metric("Lumpsum Real Value", f"₹{ls_real_value:,.0f}", 
                                         f"-₹{(ls_final_value - ls_real_value):,.0f} erosion")
                            col11.metric("SIP Real Value", f"₹{sip_real_value:,.0f}", 
                                         f"-₹{(sip_final_value - sip_real_value):,.0f} erosion")
                            real_ls_profit = ls_real_value - total_lumpsum
                            real_sip_profit = sip_real_value - sip_total_invested
                            if real_sip_profit > real_ls_profit:
                                col12.metric("Winner (Real)", "SIP", f"+₹{(real_sip_profit - real_ls_profit):,.0f}")
                            else:
                                col12.metric("Winner (Real)", "Lumpsum", f"+₹{(real_ls_profit - real_sip_profit):,.0f}")

                            # --- Step-Up SIP Info ---
                            if annual_step_up_pct > 0:
                                final_sip_amt = monthly_sip * ((1 + annual_step_up_pct / 100) ** int((months - 1) // 12))
                                st.info(f"📈 **Step-Up SIP Active:** Started at ₹{monthly_sip:,.0f}/month → Final year ₹{final_sip_amt:,.0f}/month ({annual_step_up_pct}% annual increase). Total invested: ₹{sip_total_invested:,.0f}")

                            # --- Graphing Data Prep ---
                            df_filtered_copy = df_filtered.copy()
                            df_filtered_copy[f'{fund_name} (Lumpsum)'] = df_filtered_copy['nav'] * ls_units
                            
                            # Build cumulative units for graph
                            monthly_df_graph = monthly_df.copy()
                            monthly_df_graph['units_bought'] = units_per_month
                            monthly_df_graph['cumulative_units'] = monthly_df_graph['units_bought'].cumsum()
                            
                            df_graph = df_filtered_copy.copy()
                            df_graph['sip_units'] = monthly_df_graph['cumulative_units']
                            df_graph['sip_units'] = df_graph['sip_units'].ffill().fillna(0)
                            df_graph[f'{fund_name} (SIP)'] = df_graph['sip_units'] * df_graph['nav']

                            cols_to_keep = [f'{fund_name} (Lumpsum)', f'{fund_name} (SIP)']
                            if master_graph_data.empty:
                                master_graph_data = df_graph[cols_to_keep]
                            else:
                                master_graph_data = master_graph_data.join(df_graph[cols_to_keep], how='outer')

                            # ===== PHASE 2: ROLLING RETURNS =====
                            with st.expander("📊 Rolling Returns Analysis", expanded=False):
                                # Use full historical data for richer rolling returns
                                rolling_df, r_summary = calculate_rolling_returns(df, window_years=rolling_window)
                                if r_summary:
                                    rc1, rc2, rc3, rc4 = st.columns(4)
                                    rc1.metric(f"Avg {rolling_window}Y CAGR", f"{r_summary['avg']:.2f}%")
                                    rc2.metric("Best Period", f"{r_summary['max']:.2f}%", 
                                               f"from {r_summary['max_date'].strftime('%b %Y')}")
                                    rc3.metric("Worst Period", f"{r_summary['min']:.2f}%", 
                                               f"from {r_summary['min_date'].strftime('%b %Y')}")
                                    rc4.metric("% Positive Periods", f"{r_summary['positive_pct']:.1f}%",
                                               f"of {r_summary['total_periods']} periods")
                                    
                                    # Rolling returns chart
                                    fig_rolling = go.Figure()
                                    fig_rolling.add_trace(go.Scatter(
                                        x=rolling_df['date'], y=rolling_df['cagr'],
                                        mode='lines', name=f'{rolling_window}Y Rolling CAGR',
                                        line=dict(color='#6366f1', width=1.5),
                                        fill='tozeroy',
                                        fillcolor='rgba(99,102,241,0.15)'
                                    ))
                                    fig_rolling.add_hline(y=r_summary['avg'], line_dash="dash", 
                                                          line_color="orange", annotation_text=f"Avg: {r_summary['avg']:.1f}%")
                                    fig_rolling.add_hline(y=0, line_color="red", line_width=1)
                                    fig_rolling.update_layout(
                                        title=f"{rolling_window}-Year Rolling CAGR Over Time",
                                        xaxis_title="Start Date of Period",
                                        yaxis_title="CAGR (%)",
                                        hovermode="x unified",
                                        height=400
                                    )
                                    st.plotly_chart(fig_rolling, use_container_width=True)
                                else:
                                    st.warning(f"Insufficient data for {rolling_window}-year rolling returns. Need at least {rolling_window}+ years of history.")

                            # ===== PHASE 2: DRAWDOWN VISUALIZER =====
                            with st.expander("📉 Drawdown Analysis", expanded=False):
                                dd_df, max_dd = calculate_drawdown(df_filtered)
                                
                                dc1, dc2, dc3 = st.columns(3)
                                dc1.metric("Max Drawdown", f"{max_dd['max_drawdown_pct']:.2f}%")
                                if max_dd['peak_date'] and max_dd['drawdown_date']:
                                    dc2.metric("Peak → Bottom", 
                                               f"{max_dd['peak_date'].strftime('%d %b %Y')} → {max_dd['drawdown_date'].strftime('%d %b %Y')}")
                                if max_dd['recovery_date']:
                                    dc3.metric("Recovery Time", f"{max_dd['recovery_days']} days",
                                               f"Recovered {max_dd['recovery_date'].strftime('%d %b %Y')}")
                                else:
                                    dc3.metric("Recovery", "NOT YET", "Still underwater")

                                # Drawdown chart
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=dd_df.index, y=dd_df['drawdown_pct'],
                                    mode='lines', name='Drawdown %',
                                    line=dict(color='#ef4444', width=1.5),
                                    fill='tozeroy',
                                    fillcolor='rgba(239,68,68,0.2)'
                                ))
                                fig_dd.update_layout(
                                    title="Drawdown From Peak",
                                    xaxis_title="Date",
                                    yaxis_title="Drawdown (%)",
                                    hovermode="x unified",
                                    height=350
                                )
                                st.plotly_chart(fig_dd, use_container_width=True)

                            # ===== PHASE 2: GOAL-BASED CALCULATOR =====
                            with st.expander("🎯 Goal-Based Reverse Calculator", expanded=False):
                                # Use Lumpsum CAGR as the projected rate
                                fund_cagr = ls_xirr if ls_xirr and ls_xirr > 0 else None
                                if fund_cagr:
                                    goal = goal_reverse_calculator(goal_target, goal_years, fund_cagr)
                                    
                                    st.markdown(f"**Goal:** ₹{goal_target:,.0f} in **{goal_years} years** using this fund's **{fund_cagr:.2f}% CAGR**")
                                    
                                    gc1, gc2 = st.columns(2)
                                    gc1.metric("Required One-Time Lumpsum", f"₹{goal['required_lumpsum']:,.0f}")
                                    gc2.metric("Required Monthly SIP", f"₹{goal['required_monthly_sip']:,.0f}")
                                    
                                    # Show a helpful table for multiple horizons
                                    goal_rows = []
                                    for yr in [5, 10, 15, 20, 25]:
                                        g = goal_reverse_calculator(goal_target, yr, fund_cagr)
                                        goal_rows.append({
                                            'Horizon': f'{yr} Years',
                                            'Required Lumpsum': f"₹{g['required_lumpsum']:,.0f}",
                                            'Required Monthly SIP': f"₹{g['required_monthly_sip']:,.0f}"
                                        })
                                    st.markdown(f"**Sensitivity Table** — How much you need for ₹{goal_target/10000000:.1f} Cr at {fund_cagr:.1f}% CAGR:")
                                    st.dataframe(pd.DataFrame(goal_rows).set_index('Horizon'), use_container_width=True)
                                else:
                                    st.warning("Cannot compute goal — Fund CAGR is unavailable or negative for this period.")

                            # --- 3. OPTIMAL SIP ANALYZER (Phase 1) ---
                            with st.expander("🤖 Open Optimal SIP Analyzer Results"):
                                max_sip_total = total_lumpsum * multiplier_input
                                max_monthly_sip = max_sip_total / months
                                
                                optimal_sip = None
                                optimal_profit = None
                                optimal_total = None
                                
                                for test_sip in range(500, int(max_monthly_sip) + 500, 500):
                                    test_total_invested = test_sip * months
                                    test_installment_after_tax = test_sip - (test_sip * 0.00005)
                                    test_total_units = 0
                                    
                                    for _, row in monthly_df.iterrows():
                                        test_total_units += round(test_installment_after_tax / row['nav'], 3)
                                            
                                    test_profit = (test_total_units * end_nav) - test_total_invested
                                    
                                    if test_profit > ls_profit:
                                        optimal_sip = test_sip
                                        optimal_profit = test_profit
                                        optimal_total = test_total_invested
                                        break 

                                if optimal_sip:
                                    st.success(f"**Target to Beat:** ₹{ls_profit:,.0f} (Lumpsum Profit)")
                                    st.write(f"**Optimal Monthly SIP:** ₹{optimal_sip:,.0f} / month")
                                    st.write(f"**Resulting SIP Profit:** ₹{optimal_profit:,.0f}")
                                    
                                    if optimal_total <= total_lumpsum:
                                        st.info(f"Excellent Capital Efficiency! You beat Lumpsum profit while investing less total capital.")
                                    else:
                                        extra_cost = optimal_total - total_lumpsum
                                        st.warning(f"Fair Capital Efficiency. You beat Lumpsum, but had to invest ₹{extra_cost:,.0f} more total capital to do it.")
                                else:
                                    st.error(f"No SIP strategy within a {multiplier_input}x budget (Max ₹{max_sip_total:,.0f}) could beat the Lumpsum profit for this specific timeframe.")
                        else:
                            st.write("Timeframe too short for SIP.")

            # ===== PHASE 2: CORRELATION MATRIX (Multi-Fund) =====
            if len(correlation_dfs) >= 2:
                st.divider()
                st.subheader("🔗 Portfolio Overlap / Correlation Analysis")
                
                corr_matrix, overlap_warnings = calculate_correlation_matrix(correlation_dfs)
                
                if not corr_matrix.empty:
                    # Warnings first
                    for warning in overlap_warnings:
                        st.warning(warning)
                    
                    if not overlap_warnings:
                        st.success("✅ Good diversification! No high-overlap pairs detected (all correlations below 85%).")
                    
                    # Truncate long fund names for display
                    short_names = [n[:30] + "..." if len(n) > 30 else n for n in corr_matrix.columns]
                    
                    # Heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=short_names,
                        y=short_names,
                        colorscale='RdBu_r',
                        zmin=-1, zmax=1,
                        text=corr_matrix.round(3).values,
                        texttemplate="%{text:.3f}",
                        textfont={"size": 12},
                        hoverongaps=False
                    ))
                    fig_corr.update_layout(
                        title="Daily Returns Correlation Matrix",
                        height=450,
                        xaxis=dict(tickangle=45)
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.caption("**Interpretation:** Values close to +1.0 mean the funds move together (high overlap). "
                               "Values near 0 mean they are independent (good diversification). "
                               "Values near -1.0 mean they move oppositely (natural hedge).")

            # --- Master Graph UI ---
            if not master_graph_data.empty:
                st.divider()
                st.subheader("📈 Interactive Portfolio Growth")
                
                master_graph_data = master_graph_data.ffill() 
                fig = px.line(
                    master_graph_data, 
                    x=master_graph_data.index, 
                    y=master_graph_data.columns,
                    labels={'value': 'Portfolio Value (₹)', 'date': 'Date', 'variable': 'Investment Strategy'}
                )
                fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not process data. Check the scheme codes and dates.")