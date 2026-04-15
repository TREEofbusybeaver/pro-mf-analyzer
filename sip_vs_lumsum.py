from mftool import Mftool
import pandas as pd
import sys
from scipy.optimize import brentq
from datetime import datetime

# Force UTF-8 for the Rupee symbol in terminal
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_xirr(cashflows, dates):
    """
    Calculate XIRR (Extended Internal Rate of Return).
    cashflows: list of amounts (negative = outflow, positive = inflow)
    dates: list of datetime objects corresponding to each cashflow
    Returns annualized rate as a percentage, or None if it can't converge.
    """
    if not cashflows or len(cashflows) != len(dates):
        return None
    
    day_counts = [(d - dates[0]).days for d in dates]
    
    def npv(rate):
        return sum(cf / (1 + rate) ** (dc / 365.0) for cf, dc in zip(cashflows, day_counts))
    
    try:
        rate = brentq(npv, -0.99, 10.0, maxiter=1000)
        return rate * 100  # Return as percentage
    except (ValueError, RuntimeError):
        return None


def calculate_ltcg_tax(profit, exemption=125000, rate=0.125):
    """
    Calculate India's LTCG tax on equity mutual funds (Budget 2024 rules).
    - 12.5% tax on gains above ₹1.25 Lakh exemption.
    Returns (tax_amount, post_tax_profit).
    """
    if profit <= 0:
        return 0, profit
    taxable = max(0, profit - exemption)
    tax = taxable * rate
    return tax, profit - tax


def inflation_adjusted_value(nominal_value, inflation_rate, years):
    """
    Discount a nominal future value to today's purchasing power.
    """
    if years <= 0 or inflation_rate <= 0:
        return nominal_value
    return nominal_value / ((1 + inflation_rate / 100) ** years)


def compare_strategies(scheme_code, start_date, end_date, total_lumpsum, monthly_sip,
                       annual_step_up_pct=0, inflation_rate=6.0):
    """
    Compare Lumpsum vs SIP strategies with:
    - XIRR calculation
    - LTCG Tax Impact
    - Inflation Adjustment
    - Step-Up SIP support
    """
    mf = Mftool()
    
    print(f"Fetching data for scheme {scheme_code}...")
    data = mf.get_scheme_historical_nav(scheme_code, as_json=False)
    
    if not data or 'data' not in data:
        print("Data not found. Please check the scheme code.")
        return
    
    # Look up the name in the master AMFI dictionary instead of the historical data
    all_schemes = mf.get_scheme_codes()
    fund_name = all_schemes.get(str(scheme_code), 'Unknown Fund Name')

    # Format Data into a Pandas DataFrame
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = pd.to_numeric(df['nav'])
    df = df.sort_values('date').set_index('date')
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]
    
    if df_filtered.empty:
        print("No data available for these dates.")
        return

    start_nav = df_filtered.iloc[0]['nav']
    end_nav = df_filtered.iloc[-1]['nav']
    actual_start_date = df_filtered.index[0]
    actual_end_date = df_filtered.index[-1]
    investment_years = (actual_end_date - actual_start_date).days / 365.25

    # ==========================================
    # 1. LUMPSUM CALCULATION
    # ==========================================
    ls_invested_after_tax = total_lumpsum - (total_lumpsum * 0.00005) # Stamp duty
    ls_units = round(ls_invested_after_tax / start_nav, 3) 
    ls_final_value = ls_units * end_nav
    ls_profit = ls_final_value - total_lumpsum
    ls_return_pct = (ls_profit / total_lumpsum) * 100 if total_lumpsum > 0 else 0

    # Lumpsum XIRR
    ls_cashflows = [-total_lumpsum, ls_final_value]
    ls_dates = [actual_start_date.to_pydatetime(), actual_end_date.to_pydatetime()]
    ls_xirr = calculate_xirr(ls_cashflows, ls_dates)

    # Lumpsum Tax
    ls_tax, ls_post_tax_profit = calculate_ltcg_tax(ls_profit)
    ls_post_tax_value = total_lumpsum + ls_post_tax_profit

    # Lumpsum Inflation-Adjusted
    ls_real_value = inflation_adjusted_value(ls_final_value, inflation_rate, investment_years)

    # ==========================================
    # 2. SIP CALCULATION (with Step-Up)
    # ==========================================
    monthly_data = df_filtered.resample('MS').first() 
    months = len(monthly_data)
    
    if months == 0:
        print("Timeframe too short for SIP.")
        return
    
    sip_total_invested = 0
    sip_total_units = 0
    sip_cashflows = []
    sip_dates = []
    
    current_sip = monthly_sip
    sip_start_date = monthly_data.index[0]
    
    for i, (index, row) in enumerate(monthly_data.iterrows()):
        # Step-Up: increase SIP at the start of each new year
        if annual_step_up_pct > 0 and i > 0:
            months_elapsed = (index.year - sip_start_date.year) * 12 + (index.month - sip_start_date.month)
            year_number = months_elapsed // 12
            current_sip = monthly_sip * ((1 + annual_step_up_pct / 100) ** year_number)
        
        installment_after_stamp = current_sip - (current_sip * 0.00005)
        sip_total_invested += current_sip
        
        if pd.notna(row['nav']):
            sip_total_units += round(installment_after_stamp / row['nav'], 3)
            sip_cashflows.append(-current_sip)
            sip_dates.append(index.to_pydatetime())
            
    sip_final_value = sip_total_units * end_nav
    sip_profit = sip_final_value - sip_total_invested
    sip_return_pct = (sip_profit / sip_total_invested) * 100 if sip_total_invested > 0 else 0
    
    # Calculate Average Purchase NAV for SIP (Rupee Cost Averaging metric)
    sip_avg_nav = sip_total_invested / sip_total_units if sip_total_units > 0 else 0

    # SIP XIRR
    sip_cashflows.append(sip_final_value)
    sip_dates.append(actual_end_date.to_pydatetime())
    sip_xirr = calculate_xirr(sip_cashflows, sip_dates)

    # SIP Tax
    sip_tax, sip_post_tax_profit = calculate_ltcg_tax(sip_profit)
    sip_post_tax_value = sip_total_invested + sip_post_tax_profit

    # SIP Inflation-Adjusted
    sip_real_value = inflation_adjusted_value(sip_final_value, inflation_rate, investment_years)

    # ==========================================
    # 3. DETAILED OUTPUT PRINTOUT
    # ==========================================
    print("\n" + "="*65)
    print(f"FUND: {fund_name}")
    print(f"TIMEFRAME: {actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')} ({months} Months / {investment_years:.1f} Years)")
    print("="*65)
    
    print("\n[ LUMPSUM PORTFOLIO ]")
    print(f"  Total Invested:           ₹{total_lumpsum:,.2f}")
    print(f"  Purchase NAV:             ₹{start_nav}")
    print(f"  Units Allotted:           {ls_units}")
    print(f"  Final Value:              ₹{ls_final_value:,.2f}")
    print(f"  Absolute Profit:          ₹{ls_profit:,.2f}")
    print(f"  Absolute Return:          {ls_return_pct:.2f}%")
    print(f"  XIRR (Annualized):        {ls_xirr:.2f}%" if ls_xirr is not None else "  XIRR (Annualized):        N/A")
    print(f"  ---")
    print(f"  LTCG Tax Payable:         ₹{ls_tax:,.2f}")
    print(f"  Post-Tax Profit:          ₹{ls_post_tax_profit:,.2f}")
    print(f"  Post-Tax Final Value:     ₹{ls_post_tax_value:,.2f}")
    print(f"  ---")
    print(f"  Inflation-Adjusted Value: ₹{ls_real_value:,.2f}  (at {inflation_rate}% p.a.)")
    
    print("\n[ SIP PORTFOLIO ]")
    if annual_step_up_pct > 0:
        print(f"  Starting Installment:     ₹{monthly_sip:,.2f} (Step-Up: {annual_step_up_pct}% per year)")
        final_sip = monthly_sip * ((1 + annual_step_up_pct / 100) ** int((months - 1) // 12))
        print(f"  Final Year Installment:   ₹{final_sip:,.2f}")
    else:
        print(f"  Monthly Installment:      ₹{monthly_sip:,.2f}")
    print(f"  Total Invested:           ₹{sip_total_invested:,.2f}")
    print(f"  Average Purchase NAV:     ₹{sip_avg_nav:.4f}")
    print(f"  Units Allotted:           {sip_total_units:.3f}")
    print(f"  Final Value:              ₹{sip_final_value:,.2f}")
    print(f"  Absolute Profit:          ₹{sip_profit:,.2f}")
    print(f"  Absolute Return:          {sip_return_pct:.2f}%")
    print(f"  XIRR (Annualized):        {sip_xirr:.2f}%" if sip_xirr is not None else "  XIRR (Annualized):        N/A")
    print(f"  ---")
    print(f"  LTCG Tax Payable:         ₹{sip_tax:,.2f}")
    print(f"  Post-Tax Profit:          ₹{sip_post_tax_profit:,.2f}")
    print(f"  Post-Tax Final Value:     ₹{sip_post_tax_value:,.2f}")
    print(f"  ---")
    print(f"  Inflation-Adjusted Value: ₹{sip_real_value:,.2f}  (at {inflation_rate}% p.a.)")
    
    print("\n" + "-"*65)
    print("[ HEAD-TO-HEAD COMPARISON ]")
    print(f"  Current Market NAV:       ₹{end_nav}")
    
    # Compare by Return Percentages
    if sip_return_pct > ls_return_pct:
        print(f"  Winner by Return %:       SIP outperformed by {sip_return_pct - ls_return_pct:.2f}%")
    elif ls_return_pct > sip_return_pct:
        print(f"  Winner by Return %:       Lumpsum outperformed by {ls_return_pct - sip_return_pct:.2f}%")
    else:
        print("  Winner by Return %:       Tie.")

    # Compare by XIRR
    if ls_xirr is not None and sip_xirr is not None:
        if sip_xirr > ls_xirr:
            print(f"  Winner by XIRR:           SIP ({sip_xirr:.2f}%) vs Lumpsum ({ls_xirr:.2f}%)")
        elif ls_xirr > sip_xirr:
            print(f"  Winner by XIRR:           Lumpsum ({ls_xirr:.2f}%) vs SIP ({sip_xirr:.2f}%)")
        else:
            print(f"  Winner by XIRR:           Tie ({ls_xirr:.2f}%)")

    # Compare by Absolute Profit
    if sip_profit > ls_profit:
        diff_money = sip_profit - ls_profit
        print(f"  Winner by Profit ₹:       SIP made ₹{diff_money:,.2f} MORE than Lumpsum.")
    elif ls_profit > sip_profit:
        diff_money = ls_profit - sip_profit
        print(f"  Winner by Profit ₹:       Lumpsum made ₹{diff_money:,.2f} MORE than SIP.")
    else:
        print("  Winner by Profit ₹:       Exact Tie.")

    # Compare by Post-Tax Profit
    if sip_post_tax_profit > ls_post_tax_profit:
        diff = sip_post_tax_profit - ls_post_tax_profit
        print(f"  Winner Post-Tax:          SIP takes home ₹{diff:,.2f} MORE.")
    elif ls_post_tax_profit > sip_post_tax_profit:
        diff = ls_post_tax_profit - sip_post_tax_profit
        print(f"  Winner Post-Tax:          Lumpsum takes home ₹{diff:,.2f} MORE.")
    else:
        print("  Winner Post-Tax:          Exact Tie.")

    # Rupee Cost Averaging Check
    if sip_avg_nav < start_nav:
        print("  Market Conditions:        Volatile/Falling (SIP successfully lowered your average buy price).")
    else:
        print("  Market Conditions:        Bullish/Rising (Lumpsum benefited from buying early before prices rose).")
    print("-"*65 + "\n")

    print("\n[ OPTIMAL SIP ANALYZER ]")
    budget_multiplier = 2.5 
    max_sip_total = total_lumpsum * budget_multiplier

    max_monthly_sip = max_sip_total / months
        
    optimal_sip = None
    optimal_profit = None
    optimal_total = None
        
    # Test SIP amounts starting from Rs 500, increasing by Rs 500 increments
    for test_sip in range(500, int(max_monthly_sip) + 500, 500):
        test_total_invested = test_sip * months
        test_installment_after_tax = test_sip - (test_sip * 0.00005)
        test_total_units = 0
        
        # Calculate units for this test SIP amount
        for _, row in monthly_data.iterrows():
            if pd.notna(row['nav']):
                test_total_units += round(test_installment_after_tax / row['nav'], 3)
                
        test_final_value = test_total_units * end_nav
        test_profit = test_final_value - test_total_invested
        
        # If this SIP amount generates more absolute profit than the Lumpsum, we found our winner
        if test_profit > ls_profit:
            optimal_sip = test_sip
            optimal_profit = test_profit
            optimal_total = test_total_invested
            break # Stop searching, we found the lowest viable amount

    if optimal_sip:
        print(f"  Target to Beat:           ₹{ls_profit:,.2f} (Lumpsum Profit)")
        print(f"  Optimal Monthly SIP:      ₹{optimal_sip:,.2f} / month")
        print(f"  Optimal SIP Profit:       ₹{optimal_profit:,.2f}")
        
        # Check capital efficiency
        if optimal_total < total_lumpsum:
            savings = total_lumpsum - optimal_total
            print(f"  Capital Efficiency:       EXCELLENT! You beat Lumpsum profit while investing ₹{savings:,.2f} LESS total capital.")
        else:
            extra_cost = optimal_total - total_lumpsum
            print(f"  Capital Efficiency:       FAIR. You beat Lumpsum profit, but had to invest ₹{extra_cost:,.2f} MORE total capital (within the {budget_multiplier * 100 - 100:.0f}% limit).")
    else:
        print(f"  Result:                   No SIP strategy within a +{budget_multiplier * 100 - 100:.0f}% budget (Max ₹{max_sip_total:,.2f}) could beat the Lumpsum profit for this timeframe.")


# Inputs: Scheme Code, Start Date, End Date, Lumpsum Amount, Monthly SIP Amount, Step-Up %, Inflation %
compare_strategies('118989', '2022-06-08', '2024-10-08', 200000, 15500, 
                   annual_step_up_pct=10, inflation_rate=6.0)