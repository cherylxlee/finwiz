import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import json

# ------------------------------
# DCF Calculator Module
# ------------------------------

def load_dcf_metrics_csv(uploaded_file=None, sample_data=None):
    """Load DCF metrics from a CSV file or use sample data."""
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None
    elif sample_data is not None:
        # Use the provided sample data
        try:
            csv_data = StringIO(sample_data)
            df = pd.read_csv(csv_data)
            return df
        except Exception as e:
            st.error(f"Error using sample data: {str(e)}")
            return None
    else:
        return None

def clean_and_prepare_data(df):
    """Clean and prepare the data for DCF analysis."""
    cleaned_df = df.copy()
    
    # Convert columns to numeric, handling any non-numeric values
    for col in cleaned_df.columns:
        if col != 'Metric':
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Set Metric as the index
    cleaned_df.set_index('Metric', inplace=True)
    
    return cleaned_df

def calculate_historical_metrics(df):
    """Calculate additional historical metrics needed for forecasting."""
    # Get most recent yr first
    years = sorted([col for col in df.columns if col != 'Metric'], reverse=True)
    historical_metrics = {}
    
    for metric in df.index:
        historical_metrics[metric] = {year: df.loc[metric, year] for year in years}
    
    # Calculate EBIT Margin (Operating Income / Revenue)
    if 'Revenue' in df.index and 'Operating_Income' in df.index:
        historical_metrics['EBIT_Margin'] = {}
        for year in years:
            revenue = df.loc['Revenue', year]
            operating_income = df.loc['Operating_Income', year]
            if pd.notna(revenue) and pd.notna(operating_income) and revenue != 0:
                historical_metrics['EBIT_Margin'][year] = operating_income / revenue
            else:
                historical_metrics['EBIT_Margin'][year] = np.nan
    
    # Calculate Depreciation Rate (Depreciation / Revenue)
    if 'Revenue' in df.index and 'Depreciation' in df.index:
        historical_metrics['Depreciation_Rate'] = {}
        for year in years:
            revenue = df.loc['Revenue', year]
            depreciation = df.loc['Depreciation', year]
            if pd.notna(revenue) and pd.notna(depreciation) and revenue != 0:
                historical_metrics['Depreciation_Rate'][year] = depreciation / revenue
            else:
                historical_metrics['Depreciation_Rate'][year] = np.nan
    
    # Calculate CapEx Rate (CapEx / Revenue)
    if 'Revenue' in df.index and 'CapEx' in df.index:
        historical_metrics['CapEx_Rate'] = {}
        for year in years:
            revenue = df.loc['Revenue', year]
            capex = df.loc['CapEx', year]
            if pd.notna(revenue) and pd.notna(capex) and revenue != 0:
                historical_metrics['CapEx_Rate'][year] = capex / revenue
            else:
                historical_metrics['CapEx_Rate'][year] = np.nan
    
    return historical_metrics

def calculate_wacc(risk_free_rate, market_risk_premium, beta, cost_of_debt, tax_rate, equity_weight, debt_weight):
    """Calculate the Weighted Average Cost of Capital (WACC)."""
    # Cost of Equity using CAPM
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    
    # WACC calculation (tax_rate is already in decimal form!)
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    
    return wacc

def forecast_financials(historical_metrics, forecast_years, growth_rates, latest_year):
    """Forecast future financial metrics based on historical data and growth assumptions."""
    forecasted_metrics = {}
    
    # Initialize with historical data
    for metric, values in historical_metrics.items():
        forecasted_metrics[metric] = values.copy()
    
    # Convert growth rates to multipliers (5% --> 1.05)
    revenue_growth_multipliers = {year: 1 + growth_rates['revenue'][i]/100 
                                for i, year in enumerate(forecast_years)}
    
    # Get most recent values for each metric
    latest_revenue = historical_metrics['Revenue'][latest_year]
    latest_ebit_margin = historical_metrics['EBIT_Margin'][latest_year]
    latest_tax_rate = historical_metrics['Tax_Rate'][latest_year]
    latest_depreciation_rate = historical_metrics['Depreciation_Rate'][latest_year]
    latest_capex_rate = historical_metrics['CapEx_Rate'][latest_year]
    
    # Apply growth assumptions over the forecast period
    for i, year in enumerate(forecast_years):
        if 'Revenue' not in forecasted_metrics:
            forecasted_metrics['Revenue'] = {}
        if 'Operating_Income' not in forecasted_metrics:
            forecasted_metrics['Operating_Income'] = {}
        if 'Tax_Rate' not in forecasted_metrics:
            forecasted_metrics['Tax_Rate'] = {}
        if 'Depreciation' not in forecasted_metrics:
            forecasted_metrics['Depreciation'] = {}
        if 'CapEx' not in forecasted_metrics:
            forecasted_metrics['CapEx'] = {}
        if 'EBIT_Margin' not in forecasted_metrics:
            forecasted_metrics['EBIT_Margin'] = {}
        if 'Depreciation_Rate' not in forecasted_metrics:
            forecasted_metrics['Depreciation_Rate'] = {}
        if 'CapEx_Rate' not in forecasted_metrics:
            forecasted_metrics['CapEx_Rate'] = {}
        
        # Calculate forecast year values
        if i == 0:
            # First forecast year based on latest actual year
            forecasted_metrics['Revenue'][year] = latest_revenue * revenue_growth_multipliers[year]
        else:
            # Subsequent years based on previous forecast year
            prev_year = forecast_years[i-1]
            forecasted_metrics['Revenue'][year] = forecasted_metrics['Revenue'][prev_year] * revenue_growth_multipliers[year]
        
        # EBIT Margin trends to target value (or stays constant if no target specified)
        if i == 0 or growth_rates['ebit_margin_target'] is None:
            ebit_margin = latest_ebit_margin
        else:
            # Gradually move toward target EBIT margin
            steps_remaining = len(forecast_years) - i
            ebit_margin = (forecasted_metrics['EBIT_Margin'][prev_year] * steps_remaining + 
                          growth_rates['ebit_margin_target']) / (steps_remaining + 1)
        
        forecasted_metrics['EBIT_Margin'][year] = ebit_margin
        forecasted_metrics['Operating_Income'][year] = forecasted_metrics['Revenue'][year] * ebit_margin
        
        # Tax rate remains constant unless specified otherwise
        forecasted_metrics['Tax_Rate'][year] = latest_tax_rate
        
        # Depreciation as percent of revenue
        forecasted_metrics['Depreciation_Rate'][year] = latest_depreciation_rate
        forecasted_metrics['Depreciation'][year] = forecasted_metrics['Revenue'][year] * latest_depreciation_rate
        
        # CapEx as percent of revenue
        forecasted_metrics['CapEx_Rate'][year] = latest_capex_rate  
        forecasted_metrics['CapEx'][year] = forecasted_metrics['Revenue'][year] * latest_capex_rate
    
    return forecasted_metrics

def calculate_free_cash_flows(forecasted_metrics, forecast_years):
    """Calculate Free Cash Flows based on forecasted financials."""
    free_cash_flows = {}
    
    for year in forecast_years:
        # Free Cash Flow = EBIT * (1 - Tax Rate) + Depreciation - CapEx - Change in Net Working Capital
        # Since we're assuming NWC = 0, the Net Working Capital term is eliminated
        
        ebit = forecasted_metrics['Operating_Income'][year]
        tax_rate = forecasted_metrics['Tax_Rate'][year]
        depreciation = forecasted_metrics['Depreciation'][year]
        capex = forecasted_metrics['CapEx'][year]
        
        nopat = ebit * (1 - tax_rate)  # Net Operating Profit After Tax
        fcf = nopat + depreciation - capex
        
        free_cash_flows[year] = fcf
    
    return free_cash_flows

def calculate_terminal_value(final_fcf, wacc, terminal_growth_rate):
    """Calculate the terminal value using the perpetuity growth formula."""
    # Terminal Value = FCF_n+1 / (WACC - g)
    # where FCF_n+1 is the FCF in the first year after the forecast period
    
    # Calculate FCF for the year after the forecast period
    fcf_n_plus_1 = final_fcf * (1 + terminal_growth_rate/100)
    
    # Calculate terminal value
    terminal_value = fcf_n_plus_1 / (wacc/100 - terminal_growth_rate/100)
    
    return terminal_value

def calculate_enterprise_value(free_cash_flows, terminal_value, wacc, forecast_years):
    """Calculate the enterprise value by discounting FCFs and terminal value."""
    discounted_cash_flows = {}
    present_value_factors = {}
    
    # Calculate present value factors for each forecast year
    for i, year in enumerate(forecast_years):
        # PV Factor = 1 / (1 + WACC)^n
        present_value_factors[year] = 1 / ((1 + wacc/100) ** (i + 1))
        
        # Discounted FCF = FCF * PV Factor
        discounted_cash_flows[year] = free_cash_flows[year] * present_value_factors[year]
    
    # Discount terminal value using the last year's PV factor
    discounted_terminal_value = terminal_value * present_value_factors[forecast_years[-1]]
    
    # Sum all discounted values to get enterprise value
    enterprise_value = sum(discounted_cash_flows.values()) + discounted_terminal_value
    
    return {
        'discounted_cash_flows': discounted_cash_flows,
        'discounted_terminal_value': discounted_terminal_value,
        'enterprise_value': enterprise_value
    }

def perform_dcf_analysis(historical_data, wacc_inputs, growth_assumptions):
    """Perform a complete DCF analysis and return the results."""
    # Extract the most recent year from historical data
    years = sorted([col for col in historical_data.columns if col != 'Metric'], reverse=True)
    latest_year = years[0]
    
    # Clean and prepare the data
    cleaned_data = clean_and_prepare_data(historical_data)
    
    # Calculate historical metrics
    historical_metrics = calculate_historical_metrics(cleaned_data)
    
    # Calculate WACC
    wacc = calculate_wacc(
        wacc_inputs['risk_free_rate'], 
        wacc_inputs['market_risk_premium'],
        wacc_inputs['beta'],
        wacc_inputs['cost_of_debt'],
        wacc_inputs['tax_rate'],
        wacc_inputs['equity_weight'],
        wacc_inputs['debt_weight']
    )
    
    # Generate forecast years (next 5 years after the latest year)
    current_year = int(latest_year)
    forecast_years = [str(current_year + i + 1) for i in range(5)]
    
    # Forecast financial metrics
    forecasted_metrics = forecast_financials(
        historical_metrics, 
        forecast_years, 
        growth_assumptions,
        latest_year
    )
    
    # Calculate free cash flows
    free_cash_flows = calculate_free_cash_flows(forecasted_metrics, forecast_years)
    
    # Calculate terminal value
    terminal_value = calculate_terminal_value(
        free_cash_flows[forecast_years[-1]], 
        wacc, 
        growth_assumptions['terminal_growth_rate']
    )
    
    # Calculate enterprise value
    valuation_results = calculate_enterprise_value(
        free_cash_flows, 
        terminal_value, 
        wacc, 
        forecast_years
    )
    
    # Compile all results
    results = {
        'historical_metrics': historical_metrics,
        'forecasted_metrics': forecasted_metrics,
        'free_cash_flows': free_cash_flows,
        'wacc': wacc,
        'terminal_value': terminal_value,
        'valuation_results': valuation_results,
        'forecast_years': forecast_years,
        'latest_year': latest_year
    }
    
    return results

def create_data_table(data_dict, years, title, format_func=None):
    """Create a formatted table from a dictionary of yearly data."""
    # Create headers
    headers = ["Metric"] + list(years)
    
    # Create rows
    rows = []
    for metric, values in data_dict.items():
        row = [metric]
        for year in years:
            value = values.get(year, "N/A")
            if format_func and pd.notna(value) and value != "N/A":
                value = format_func(value)
            row.append(value)
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    
    # Display table
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    
    return df

def create_fcf_chart(historical_metrics, forecasted_metrics, free_cash_flows, forecast_years, latest_year):
    """Create a chart showing historical and projected free cash flows."""
    # Calculate historical FCF
    historical_fcf = {}
    historical_years = sorted([y for y in historical_metrics['Revenue'].keys()])
    
    for year in historical_years:
        if (year in historical_metrics['Operating_Income'] and 
            year in historical_metrics['Tax_Rate'] and 
            year in historical_metrics['Depreciation'] and 
            year in historical_metrics['CapEx']):
            
            ebit = historical_metrics['Operating_Income'][year]
            tax_rate = historical_metrics['Tax_Rate'][year] / 100
            depreciation = historical_metrics['Depreciation'][year]
            capex = historical_metrics['CapEx'][year]
            
            if pd.notna(ebit) and pd.notna(tax_rate) and pd.notna(depreciation) and pd.notna(capex):
                nopat = ebit * (1 - tax_rate)
                historical_fcf[year] = nopat + depreciation - capex
    
    # Combine historical and forecasted FCF
    all_years = historical_years + forecast_years
    all_fcf = []
    
    for year in all_years:
        fcf_value = historical_fcf.get(year, free_cash_flows.get(year, None))
        if fcf_value is not None:
            all_fcf.append({
                'Year': year,
                'Free Cash Flow': fcf_value,
                'Type': 'Historical' if year in historical_years else 'Forecast'
            })
    
    # Create DataFrame
    fcf_df = pd.DataFrame(all_fcf)
    
    # Create chart
    fig = px.bar(
        fcf_df, 
        x='Year', 
        y='Free Cash Flow', 
        color='Type',
        title='Historical and Projected Free Cash Flows',
        labels={'Free Cash Flow': 'Free Cash Flow (in millions)'},
        color_discrete_map={'Historical': 'darkblue', 'Forecast': 'lightblue'}
    )
    
    # Add a line for Total FCF
    fig.add_trace(
        go.Scatter(
            x=fcf_df['Year'],
            y=fcf_df['Free Cash Flow'],
            mode='lines+markers',
            name='FCF Trend',
            line=dict(color='red', width=2)
        )
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Amount (in millions)',
        legend_title='Data Type',
        height=500
    )
    
    return fig

def create_valuation_waterfall_chart(dcf_results):
    """Create a waterfall chart showing the build-up to enterprise value."""
    valuation_results = dcf_results['valuation_results']
    discounted_cash_flows = valuation_results['discounted_cash_flows']
    discounted_terminal_value = valuation_results['discounted_terminal_value']
    
    # Create data for the waterfall chart
    waterfall_data = []
    
    # Add each year's DCF as a separate component
    for year, dcf in discounted_cash_flows.items():
        waterfall_data.append({
            'Component': f'DCF {year}',
            'Value': dcf,
            'Type': 'Forecast DCF'
        })
    
    # Add terminal value
    waterfall_data.append({
        'Component': 'Terminal Value (PV)',
        'Value': discounted_terminal_value,
        'Type': 'Terminal Value'
    })
    
    # Add total enterprise value
    waterfall_data.append({
        'Component': 'Enterprise Value',
        'Value': valuation_results['enterprise_value'],
        'Type': 'Total'
    })
    
    # Create DataFrame
    waterfall_df = pd.DataFrame(waterfall_data)
    
    # Create a cumulative sum for the waterfall effect
    measure = ['relative'] * (len(waterfall_df) - 1) + ['total']
    
    # Create the waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Enterprise Value Build-up",
        orientation="v",
        measure=measure,
        x=waterfall_df['Component'],
        y=waterfall_df['Value'],
        textposition="outside",
        text=waterfall_df['Value'].apply(lambda x: f"${x:,.0f}M"),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    # Improve layout
    fig.update_layout(
        title="DCF Valuation Build-up",
        showlegend=True,
        height=500
    )
    
    return fig

def dcf_calculator_ui():
    """Main UI function for the DCF Calculator tab."""
    st.subheader("DCF Calculator")
    
    st.markdown("""
    This tool performs a simple Discounted Cash Flow (DCF) analysis based on the extracted financial metrics.
    The calculator uses a transparent, step-by-step approach to forecast future cash flows and calculate
    enterprise value.
    """)
    
    # Input section
    with st.expander("Assumptions", expanded=False):
        st.markdown("""
        ### 1) Net Working Capital = 0

        For tech companies like Apple, Alphabet, Meta, Microsoft, and Tesla, assuming Net Working Capital (NWC) = 0 
        is reasonable for a simplified DCF model for several reasons:

        1. **Minimal inventory requirements**: Unlike traditional manufacturers, these tech companies (especially SaaS companies) 
           have limited physical inventory needs.
        
        2. **Favorable cash conversion cycles**: Many tech companies receive payment before delivering services 
           (subscriptions, advance payments), creating negative working capital cycles.
        
        3. **Cash-rich balance sheets**: These companies often maintain large cash reserves, offsetting working capital needs.
        
        4. **Limited account receivables**: Subscription and direct-to-consumer business models reduce outstanding 
           receivables compared to traditional businesses.
        
        5. **Relatively small working capital changes**: Year-over-year changes in NWC for these companies tend to 
           be small relative to their overall cash flows.
        
        This simplification allows for a more straightforward model while maintaining reasonable accuracy for 
        valuation purposes. For more detailed analysis, working capital could be incorporated as a separate 
        forecast component.
                    
        ### 2) Capital Structure: 80% equity, 20% debt for tech companies
        1. **Historically Low Leverage**: Most tech companies have historically maintained low debt levels relative 
           to equity. For example, Apple and Microsoft have taken on some debt primarily for tax arbitrage or buybacks—not
           out of necessity. Companies like Alphabet and Meta have virtually no long-term debt, reflecting minimal reliance on leverage.

        2. **Strong Cash Flow Generation**: These firms generate massive free cash flows that reduce their need for external financing.

        3. **High Credit Ratings**: Their strong credit ratings (often AA or better) mean they could take on more debt, but they choose not to.

        4. **Growth Orientation Over Financial Engineering**: These companies prioritize innovation, R&D, and strategic investments over aggressive
           financial structuring. They often reinvest cash into growth rather than taking on debt to fund dividends or repurchases.

        5. **Market Confidence in Equity**: Investors place a premium on their stock due to growth potential, network effects, and brand power.
        """)
    
    st.subheader("Step 1: Load Financial Data")
    
    data_source = st.radio("Select data source:", ["Upload CSV", "Use Sample Data"])
    
    sample_data = """Metric,2024,2023,2022
Revenue,383285,394328,365817
Operating_Income,113031,122063,121191
CapEx,11284,11284,10708
Depreciation,11843,11843,11104
Tax_Rate,0.132,0.147,0.164"""
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload DCF metrics CSV file", type=['csv'])
        if uploaded_file is not None:
            dcf_data = load_dcf_metrics_csv(uploaded_file=uploaded_file)
        else:
            dcf_data = None
    else:
        dcf_data = load_dcf_metrics_csv(sample_data=sample_data)
        st.info("Using sample data for Apple Inc.")
    
    # Display the loaded data
    if dcf_data is not None:
        st.write("Loaded Financial Metrics:")
        st.dataframe(dcf_data)
        
        # Input parameters for WACC
        st.subheader("Step 2: Set WACC Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_free_rate = st.number_input("Risk-Free Rate (%)", value=3.5, min_value=0.0, max_value=10.0, step=0.1)
            market_risk_premium = st.number_input("Market Risk Premium (%)", value=4.5, min_value=1.0, max_value=15.0, step=0.1)
            beta = st.number_input("Company Beta", value=1.2, min_value=0.1, max_value=3.0, step=0.1)
        
        with col2:
            cost_of_debt = st.number_input("Cost of Debt (%)", value=3.5, min_value=0.0, max_value=15.0, step=0.1)
            # Get tax rate and convert to percentage for display
            original_tax_rate = float(dcf_data.loc[dcf_data['Metric'] == 'Tax_Rate', '2024'].values[0])
            display_tax_rate = original_tax_rate * 100 if original_tax_rate < 1 else original_tax_rate
            tax_rate = st.number_input("Effective Tax Rate (%)", value=display_tax_rate, min_value=0.0, max_value=50.0, step=0.1)
            
            # Capital structure
            equity_weight = st.slider("Equity Weight (%)", value=80, min_value=0, max_value=100, step=1)
            debt_weight = 100 - equity_weight
            st.write(f"Debt Weight: {debt_weight}%")
            
        # Compile WACC inputs
        wacc_inputs = {
            'risk_free_rate': risk_free_rate,
            'market_risk_premium': market_risk_premium,
            'beta': beta,
            'cost_of_debt': cost_of_debt,
            'tax_rate': tax_rate / 100,
            'equity_weight': equity_weight / 100,
            'debt_weight': debt_weight / 100
        }
        
        # Calculate and display WACC
        wacc = calculate_wacc(
            risk_free_rate, 
            market_risk_premium, 
            beta, 
            cost_of_debt, 
            tax_rate / 100,
            equity_weight / 100, 
            debt_weight / 100
        )
        
        st.info(f"Calculated WACC: {wacc:.2f}%")
        

        st.subheader("Step 3: Set Growth Assumptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate average historical revenue growth if available
            years = sorted([col for col in dcf_data.columns if col != 'Metric'], reverse=True)
            avg_revenue_growth = 0.0
            
            if len(years) >= 2:
                revenue_values = {}
                for year in years:
                    revenue_row = dcf_data.loc[dcf_data['Metric'] == 'Revenue']
                    if not revenue_row.empty:
                        revenue_values[year] = revenue_row[year].values[0]
                
                # Calculate year-over-year growth rates
                growth_rates = []
                sorted_years = sorted(revenue_values.keys())
                for i in range(1, len(sorted_years)):
                    prev_year = sorted_years[i-1]
                    curr_year = sorted_years[i]
                    if revenue_values[prev_year] > 0:
                        growth_rate = (revenue_values[curr_year] - revenue_values[prev_year]) / revenue_values[prev_year] * 100
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    avg_revenue_growth = sum(growth_rates) / len(growth_rates)
            
            st.write("Revenue Growth Rates (%)")
            growth_rates = []
            
            latest_year = int(years[0])
            
            # Create 5 forecast years
            forecast_years = [str(latest_year + i + 1) for i in range(5)]
            
            # Create a default declining growth rate (starting with historical average)
            default_growth_rate = max(7.0, avg_revenue_growth)
            default_rates = [
                default_growth_rate,
                default_growth_rate * 0.9,
                default_growth_rate * 0.8,
                default_growth_rate * 0.7,
                default_growth_rate * 0.6
            ]
            
            # Input fields for each forecast year
            for i, year in enumerate(forecast_years):
                growth_rate = st.number_input(
                    f"{year}", 
                    value=default_rates[i], 
                    min_value=-20.0, 
                    max_value=50.0, 
                    step=0.5,
                    key=f"growth_{year}"
                )
                growth_rates.append(growth_rate)
        
        with col2:
            # Get current EBIT margin
            ebit_margin = None
            revenue_row = dcf_data.loc[dcf_data['Metric'] == 'Revenue']
            operating_income_row = dcf_data.loc[dcf_data['Metric'] == 'Operating_Income']
            
            if not revenue_row.empty and not operating_income_row.empty:
                latest_revenue = revenue_row[years[0]].values[0]
                latest_operating_income = operating_income_row[years[0]].values[0]
                
                if latest_revenue > 0:
                    ebit_margin = (latest_operating_income / latest_revenue) * 100
            
            if ebit_margin is None:
                ebit_margin = 25.0
            
            st.write("Other Assumptions")
            
            # Target EBIT margin (default to current)
            use_target_margin = st.checkbox("Set Target EBIT Margin", value=False)
            if use_target_margin:
                target_ebit_margin = st.number_input(
                    "Target EBIT Margin in Year 5 (%)", 
                    value=ebit_margin, 
                    min_value=0.0, 
                    max_value=100.0, 
                    step=0.5
                )
            else:
                target_ebit_margin = None
                st.write(f"Using current EBIT Margin: {ebit_margin:.2f}%")
            
            # Terminal growth rate
            terminal_growth_rate = st.number_input(
                "Terminal Growth Rate (%)", 
                value=4.0, 
                min_value=0.0, 
                max_value=5.0, 
                step=0.1
            )
        
        # Compile growth assumptions
        growth_assumptions = {
            'revenue': growth_rates,
            'ebit_margin_target': target_ebit_margin,
            'terminal_growth_rate': terminal_growth_rate
        }
        
        # Button to run DCF analysis
        if st.button("Run DCF Analysis", type="primary"):
            with st.spinner("Calculating DCF..."):
                # Perform DCF analysis
                dcf_results = perform_dcf_analysis(dcf_data, wacc_inputs, growth_assumptions)
                
                # Display results
                st.subheader("DCF Analysis Results")
                
                # Historical and forecasted metrics
                tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Forecast Data", "Charts", "Explanation"])
                
                with tab1:
                    # Enterprise value
                    st.subheader(f"Enterprise Value: ${dcf_results['valuation_results']['enterprise_value']:,.0f} million")
                    
                    # Create a pie chart of value components
                    dcf_sum = sum(dcf_results['valuation_results']['discounted_cash_flows'].values())
                    terminal_value = dcf_results['valuation_results']['discounted_terminal_value']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Explicit Forecast Period', 'Terminal Value'],
                        values=[dcf_sum, terminal_value],
                        hole=.3
                    )])
                    
                    fig.update_layout(
                        title_text="Enterprise Value Breakdown",
                        annotations=[dict(text=f'${dcf_results["valuation_results"]["enterprise_value"]:,.0f}M', x=0.5, y=0.5, font_size=15, showarrow=False)]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add waterfall chart
                    waterfall_chart = create_valuation_waterfall_chart(dcf_results)
                    st.plotly_chart(waterfall_chart, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("WACC", f"{dcf_results['wacc']:.2f}%")
                    with col2:
                        st.metric("Terminal Growth Rate", f"{growth_assumptions['terminal_growth_rate']:.2f}%")
                    with col3:
                        tv_percent = (terminal_value / dcf_results['valuation_results']['enterprise_value']) * 100
                        st.metric("Terminal Value %", f"{tv_percent:.1f}%")
                
                with tab2:
                    # Display forecasted financials
                    forecast_years = dcf_results['forecast_years']
                    historical_years = sorted([y for y in dcf_results['historical_metrics']['Revenue'].keys()])
                    all_years = historical_years + forecast_years
                    
                    # Key financial metrics table
                    st.subheader("Key Financial Metrics")
                    
                    # Combine historical and forecasted metrics
                    combined_metrics = {
                        'Revenue': {},
                        'Revenue Growth (%)': {},
                        'Operating Income': {},
                        'EBIT Margin (%)': {},
                        'Depreciation': {},
                        'CapEx': {},
                        'Free Cash Flow': {}
                    }
                    
                    # Add historical data
                    for year in historical_years:
                        # Revenue
                        combined_metrics['Revenue'][year] = dcf_results['historical_metrics']['Revenue'].get(year)
                        
                        # Revenue Growth
                        if 'Revenue_Growth' in dcf_results['historical_metrics'] and year in dcf_results['historical_metrics']['Revenue_Growth']:
                            combined_metrics['Revenue Growth (%)'][year] = dcf_results['historical_metrics']['Revenue_Growth'].get(year)
                        else:
                            # Calculate if not available
                            if year != historical_years[0]:  # Skip first year as we need previous year data
                                prev_year = historical_years[historical_years.index(year) - 1]
                                current_rev = dcf_results['historical_metrics']['Revenue'].get(year)
                                prev_rev = dcf_results['historical_metrics']['Revenue'].get(prev_year)
                                if current_rev and prev_rev and prev_rev != 0:
                                    growth = ((current_rev - prev_rev) / prev_rev) * 100
                                    combined_metrics['Revenue Growth (%)'][year] = growth
                        
                        # Operating Income
                        combined_metrics['Operating Income'][year] = dcf_results['historical_metrics']['Operating_Income'].get(year)
                        
                        # EBIT Margin
                        if 'EBIT_Margin' in dcf_results['historical_metrics'] and year in dcf_results['historical_metrics']['EBIT_Margin']:
                            combined_metrics['EBIT Margin (%)'][year] = dcf_results['historical_metrics']['EBIT_Margin'].get(year) * 100
                        
                        # Depreciation
                        combined_metrics['Depreciation'][year] = dcf_results['historical_metrics']['Depreciation'].get(year)
                        
                        # CapEx
                        combined_metrics['CapEx'][year] = dcf_results['historical_metrics']['CapEx'].get(year)
                        
                        # Free Cash Flow (historical)
                        if all(year in dcf_results['historical_metrics'][metric] for metric in ['Operating_Income', 'Tax_Rate', 'Depreciation', 'CapEx']):
                            ebit = dcf_results['historical_metrics']['Operating_Income'][year]
                            tax_rate = dcf_results['historical_metrics']['Tax_Rate'][year] / 100
                            depreciation = dcf_results['historical_metrics']['Depreciation'][year]
                            capex = dcf_results['historical_metrics']['CapEx'][year]
                            
                            if all(pd.notna(val) for val in [ebit, tax_rate, depreciation, capex]):
                                nopat = ebit * (1 - tax_rate)
                                fcf = nopat + depreciation - capex
                                combined_metrics['Free Cash Flow'][year] = fcf
                    
                    # Add forecasted data
                    for year in forecast_years:
                        # Revenue
                        combined_metrics['Revenue'][year] = dcf_results['forecasted_metrics']['Revenue'].get(year)
                        
                        # Revenue Growth (calculated from forecasted data)
                        year_idx = forecast_years.index(year)
                        combined_metrics['Revenue Growth (%)'][year] = growth_assumptions['revenue'][year_idx]
                        
                        # Operating Income
                        combined_metrics['Operating Income'][year] = dcf_results['forecasted_metrics']['Operating_Income'].get(year)
                        
                        # EBIT Margin
                        if 'EBIT_Margin' in dcf_results['forecasted_metrics'] and year in dcf_results['forecasted_metrics']['EBIT_Margin']:
                            combined_metrics['EBIT Margin (%)'][year] = dcf_results['forecasted_metrics']['EBIT_Margin'].get(year) * 100
                        
                        # Depreciation
                        combined_metrics['Depreciation'][year] = dcf_results['forecasted_metrics']['Depreciation'].get(year)
                        
                        # CapEx
                        combined_metrics['CapEx'][year] = dcf_results['forecasted_metrics']['CapEx'].get(year)
                        
                        # Free Cash Flow (forecasted)
                        combined_metrics['Free Cash Flow'][year] = dcf_results['free_cash_flows'].get(year)
                    
                    # Create a formatted table with the combined metrics
                    combined_df = pd.DataFrame(index=combined_metrics.keys(), columns=all_years)
                    
                    for metric, values in combined_metrics.items():
                        for year in all_years:
                            combined_df.loc[metric, year] = values.get(year, None)
                    
                    # Format the table
                    formatted_df = combined_df.copy()
                    
                    # Apply formatting based on metric type
                    for idx in formatted_df.index:
                        for col in formatted_df.columns:
                            value = formatted_df.loc[idx, col]
                            if pd.isna(value) or value is None:
                                formatted_df.loc[idx, col] = "N/A"
                            elif "%" in idx:  # Percentage metrics
                                formatted_df.loc[idx, col] = f"{value:.2f}%"
                            else:  # Dollar amount metrics
                                formatted_df.loc[idx, col] = f"${value:,.0f}M"
                    
                    # Display the table
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # DCF Calculation table
                    st.subheader("DCF Calculation")
                    
                    # Create a DCF calculation table
                    dcf_calc = pd.DataFrame(index=[
                        'Free Cash Flow',
                        'Discount Factor',
                        'Present Value',
                        'Cumulative PV'
                    ], columns=forecast_years)
                    
                    cumulative_pv = 0
                    for year in forecast_years:
                        # Get values from results
                        fcf = dcf_results['free_cash_flows'].get(year)
                        discount_factor = 1 / ((1 + dcf_results['wacc']/100) ** (forecast_years.index(year) + 1))
                        pv = fcf * discount_factor
                        cumulative_pv += pv
                        
                        # Add to table
                        dcf_calc.loc['Free Cash Flow', year] = fcf
                        dcf_calc.loc['Discount Factor', year] = discount_factor
                        dcf_calc.loc['Present Value', year] = pv
                        dcf_calc.loc['Cumulative PV', year] = cumulative_pv
                    
                    # Add terminal value row
                    last_year = forecast_years[-1]
                    terminal_value_row = pd.Series(dtype='float64', index=forecast_years)
                    terminal_value_row[last_year] = dcf_results['terminal_value']
                    dcf_calc.loc['Terminal Value', :] = terminal_value_row
                    
                    terminal_pv_row = pd.Series(dtype='float64', index=forecast_years)
                    last_discount_factor = dcf_calc.loc['Discount Factor', last_year]
                    terminal_pv_row[last_year] = dcf_results['terminal_value'] * last_discount_factor
                    dcf_calc.loc['Terminal Value (PV)', :] = terminal_pv_row
                    
                    # Format the DCF table
                    formatted_dcf = dcf_calc.copy()
                    
                    for idx in formatted_dcf.index:
                        for col in formatted_dcf.columns:
                            value = formatted_dcf.loc[idx, col]
                            if pd.isna(value) or value is None or value == 0:
                                formatted_dcf.loc[idx, col] = "N/A"
                            elif idx == 'Discount Factor':
                                formatted_dcf.loc[idx, col] = f"{value:.4f}"
                            else:  # Dollar amount metrics
                                formatted_dcf.loc[idx, col] = f"${value:,.0f}M"
                    
                    # Add enterprise value
                    enterprise_value = dcf_results['valuation_results']['enterprise_value']
                    formatted_dcf.loc['Enterprise Value', :] = ["N/A"] * (len(forecast_years) - 1) + [f"${enterprise_value:,.0f}M"]
                    
                    # Display the DCF calculation table
                    st.dataframe(formatted_dcf, use_container_width=True)
                
                with tab3:
                    # Create charts
                    st.subheader("Financial Projections")
                    
                    # Cash Flow Chart
                    fcf_chart = create_fcf_chart(
                        dcf_results['historical_metrics'],
                        dcf_results['forecasted_metrics'],
                        dcf_results['free_cash_flows'],
                        forecast_years,
                        historical_years[0]
                    )
                    st.plotly_chart(fcf_chart, use_container_width=True)
                    
                    # Revenue and Operating Income Chart
                    revenue_oi_data = []
                    
                    # Add historical data
                    for year in historical_years:
                        if year in dcf_results['historical_metrics']['Revenue'] and year in dcf_results['historical_metrics']['Operating_Income']:
                            revenue = dcf_results['historical_metrics']['Revenue'][year]
                            operating_income = dcf_results['historical_metrics']['Operating_Income'][year]
                            
                            if pd.notna(revenue) and pd.notna(operating_income):
                                revenue_oi_data.append({
                                    'Year': year,
                                    'Revenue': revenue,
                                    'Operating Income': operating_income,
                                    'Type': 'Historical'
                                })
                    
                    # Add forecasted data
                    for year in forecast_years:
                        if year in dcf_results['forecasted_metrics']['Revenue'] and year in dcf_results['forecasted_metrics']['Operating_Income']:
                            revenue = dcf_results['forecasted_metrics']['Revenue'][year]
                            operating_income = dcf_results['forecasted_metrics']['Operating_Income'][year]
                            
                            revenue_oi_data.append({
                                'Year': year,
                                'Revenue': revenue,
                                'Operating Income': operating_income,
                                'Type': 'Forecast'
                            })
                    
                    # Create DataFrame
                    revenue_oi_df = pd.DataFrame(revenue_oi_data)
                    
                    # Create chart
                    fig = go.Figure()
                    
                    # Add Revenue bars
                    fig.add_trace(go.Bar(
                        x=revenue_oi_df['Year'],
                        y=revenue_oi_df['Revenue'],
                        name='Revenue',
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    
                    # Add Operating Income bars
                    fig.add_trace(go.Bar(
                        x=revenue_oi_df['Year'],
                        y=revenue_oi_df['Operating Income'],
                        name='Operating Income',
                        marker_color='darkblue',
                        opacity=0.9
                    ))
                    
                    # Add EBIT Margin line on secondary y-axis
                    ebit_margin_data = []
                    
                    # Add historical EBIT margins
                    for year in historical_years:
                        if year in dcf_results['historical_metrics']['EBIT_Margin']:
                            ebit_margin = dcf_results['historical_metrics']['EBIT_Margin'][year] * 100
                            
                            if pd.notna(ebit_margin):
                                ebit_margin_data.append({
                                    'Year': year,
                                    'EBIT Margin': ebit_margin,
                                    'Type': 'Historical'
                                })
                    
                    # Add forecasted EBIT margins
                    for year in forecast_years:
                        if year in dcf_results['forecasted_metrics']['EBIT_Margin']:
                            ebit_margin = dcf_results['forecasted_metrics']['EBIT_Margin'][year] * 100
                            
                            ebit_margin_data.append({
                                'Year': year,
                                'EBIT Margin': ebit_margin,
                                'Type': 'Forecast'
                            })
                    
                    # Create DataFrame for EBIT Margin
                    ebit_margin_df = pd.DataFrame(ebit_margin_data)
                    
                    # Add EBIT Margin line
                    fig.add_trace(go.Scatter(
                        x=ebit_margin_df['Year'],
                        y=ebit_margin_df['EBIT Margin'],
                        name='EBIT Margin (%)',
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        yaxis='y2'
                    ))
                    
                    # Update layout with secondary y-axis
                    fig.update_layout(
                        title='Revenue and Operating Income Projection',
                        xaxis_title='Year',
                        yaxis=dict(
                            title='Amount (in millions)',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='EBIT Margin (%)',
                            side='right',
                            overlaying='y',
                            range=[0, 50]
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a DCF Sensitivity Analysis
                    st.subheader("Sensitivity Analysis")
                    
                    # Create sensitivity analysis data
                    wacc_range = np.arange(max(0.5, dcf_results['wacc'] - 3), dcf_results['wacc'] + 3.1, 0.5)
                    growth_range = np.arange(max(0.5, growth_assumptions['terminal_growth_rate'] - 1.5), 
                                         growth_assumptions['terminal_growth_rate'] + 1.6, 0.5)
                    
                    # Create empty DataFrame for sensitivity analysis
                    sensitivity_df = pd.DataFrame(index=wacc_range, columns=growth_range)
                    
                    # Calculate enterprise value for each combination
                    for wacc_val in wacc_range:
                        for growth_val in growth_range:
                            # Recalculate terminal value
                            final_fcf = dcf_results['free_cash_flows'][forecast_years[-1]]
                            fcf_n_plus_1 = final_fcf * (1 + growth_val/100)
                            new_terminal_value = fcf_n_plus_1 / (wacc_val/100 - growth_val/100)
                            
                            # Recalculate present value factors
                            new_discounted_cash_flows = {}
                            for i, year in enumerate(forecast_years):
                                pv_factor = 1 / ((1 + wacc_val/100) ** (i + 1))
                                new_discounted_cash_flows[year] = dcf_results['free_cash_flows'][year] * pv_factor
                            
                            # Discount terminal value
                            last_pv_factor = 1 / ((1 + wacc_val/100) ** len(forecast_years))
                            new_discounted_terminal_value = new_terminal_value * last_pv_factor
                            
                            # Calculate new enterprise value
                            new_enterprise_value = sum(new_discounted_cash_flows.values()) + new_discounted_terminal_value
                            
                            # Add to sensitivity DataFrame
                            sensitivity_df.loc[wacc_val, growth_val] = new_enterprise_value
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=sensitivity_df.values,
                        x=sensitivity_df.columns,
                        y=sensitivity_df.index,
                        colorscale='Blues',
                        hoverongaps = False,
                        hovertemplate='WACC: %{y}%<br>Terminal Growth: %{x}%<br>Enterprise Value: $%{z:,.0f}M<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Enterprise Value Sensitivity Analysis',
                        xaxis_title='Terminal Growth Rate (%)',
                        yaxis_title='WACC (%)',
                        height=600
                    )
                    
                    # Add annotations for selected values
                    fig.add_annotation(
                        x=growth_assumptions['terminal_growth_rate'],
                        y=dcf_results['wacc'],
                        text=f"Selected<br>${dcf_results['valuation_results']['enterprise_value']:,.0f}M",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.subheader("DCF Methodology Explanation")
                    
                    st.markdown("""
                    ### How the DCF Model Works
                    
                    This DCF model follows a standard approach to business valuation:
                    
                    #### 1. Forecast Free Cash Flows
                    The model forecasts future free cash flows (FCF) based on historical financial data and growth assumptions.
                    
                    **FCF = EBIT × (1 - Tax Rate) + Depreciation - CapEx**
                    
                    Net working capital is assumed to be zero for simplicity, which is reasonable for many tech companies.
                    
                    #### 2. Calculate Discount Rate (WACC)
                    The Weighted Average Cost of Capital (WACC) is calculated using:
                    
                    **WACC = (Weight of Equity × Cost of Equity) + (Weight of Debt × Cost of Debt × (1 - Tax Rate))**
                    
                    Where Cost of Equity is calculated using the Capital Asset Pricing Model (CAPM):
                    
                    **Cost of Equity = Risk-Free Rate + Beta × Market Risk Premium**
                    
                    #### 3. Discount Future Cash Flows
                    Each forecasted FCF is discounted to present value using the WACC:
                    
                    **Present Value of FCF = FCF / (1 + WACC)^n**
                    
                    Where n is the number of years from now.
                    
                    #### 4. Calculate Terminal Value
                    The terminal value represents all cash flows beyond the explicit forecast period, calculated using the 
                    perpetuity growth model:
                    
                    **Terminal Value = FCF_n+1 / (WACC - Terminal Growth Rate)**
                    
                    Where FCF_n+1 is the cash flow in the first year after the forecast period.
                    
                    #### 5. Calculate Enterprise Value
                    The enterprise value is the sum of all discounted FCFs plus the discounted terminal value:
                    
                    **Enterprise Value = Sum of Discounted FCFs + Discounted Terminal Value**
                    
                    ### Simplifications in This Model
                    
                    1. **Net Working Capital = 0**: As discussed earlier, this assumption works well for tech companies.
                    
                    2. **No Debt Adjustment**: Enterprise Value is not adjusted for debt to get to Equity Value.
                    
                    3. **Linear Growth Projections**: The model uses simple growth rates rather than detailed business forecasting.
                    
                    4. **Constant EBIT Margins**: The model assumes relatively stable margins unless a target is specified.                    
                    """)
                
                # Export options
                st.subheader("Export DCF Results")
                
                # Create a JSON representation of the DCF results
                export_data = {
                    'inputs': {
                        'wacc_inputs': wacc_inputs,
                        'growth_assumptions': growth_assumptions,
                        'historical_data': dcf_data.to_dict()
                    },
                    'results': {
                        'enterprise_value': dcf_results['valuation_results']['enterprise_value'],
                        'wacc': dcf_results['wacc'],
                        'terminal_value': dcf_results['terminal_value'],
                        'discounted_terminal_value': dcf_results['valuation_results']['discounted_terminal_value'],
                        'discounted_cash_flows': dcf_results['valuation_results']['discounted_cash_flows'],
                        'free_cash_flows': dcf_results['free_cash_flows'],
                        'forecasted_metrics': {
                            metric: {str(year): value for year, value in values.items()}
                            for metric, values in dcf_results['forecasted_metrics'].items()
                        }
                    }
                }
                
                # Convert to JSON string
                json_str = json.dumps(export_data, indent=2)
                
                # Download button
                st.download_button(
                    "Download DCF Results (JSON)",
                    json_str,
                    "dcf_results.json",
                    "application/json",
                    key='download-json'
                )
