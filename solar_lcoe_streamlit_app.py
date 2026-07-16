
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(
    page_title="Solar LCOE Calculator",
    page_icon="☀️",
    layout="wide"
)

# Light visual customisation for the app and data tables
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }
    .stApp {
        background: #ffffff;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    .ie-hero {
        background: linear-gradient(135deg, #0d3b66 0%, #1f6f9f 100%);
        color: white;
        border-radius: 0 0 18px 18px;
        padding: 2.2rem 2.4rem;
        margin: -1.25rem -1rem 1.8rem -1rem;
        border-bottom: 6px solid #dbeaf4;
    }
    .ie-brand {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
        opacity: 0.95;
    }
    .ie-hero h1 {
        color: white;
        margin: 0 0 0.45rem 0;
        font-size: 2.25rem;
    }
    .ie-hero p {
        color: white;
        margin: 0;
        max-width: 850px;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        background: #f7fafc;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0d3b66;
    }
    .comparison-banner {
        border: 1px solid #c9dce9;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.25rem 0 1rem 0;
        background: #eef6fb;
        text-align: center;
    }
    .comparison-banner strong {
        font-size: 1.2rem;
        color: #0d3b66;
    }
    .methodology-box {
        background: #f4f8fb;
        border-left: 5px solid #1f6f9f;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin: 0.75rem 0 1.25rem 0;
        line-height: 1.6;
    }
    .methodology-box h3 {
        color: #0d3b66;
        margin-top: 0;
    }
    .nav-heading {
        text-align: center;
        color: #0d3b66;
        margin-top: 0.5rem;
    }
    div.stButton > button, div[data-testid="stLinkButton"] > a {
        border-radius: 8px;
        font-weight: 700;
    }
    .model-cta {
        background: #0d3b66;
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0 1.25rem 0;
        text-align: center;
        color: white;
    }
    .model-cta h3 {
        color: white;
        margin: 0 0 0.8rem 0;
        font-size: 2rem;
        line-height: 1.25;
        font-weight: 800;
        text-align: center;
    }
    .model-cta p {
        color: white;
        margin: 0 auto 1.2rem auto;
        max-width: 920px;
        font-size: 1.15rem;
        line-height: 1.7;
        text-align: center;
    }
    .model-cta a {
        display: inline-block;
        background: white;
        color: #0d3b66 !important;
        text-decoration: none;
        font-weight: 800;
        font-size: 1.08rem;
        padding: 0.9rem 1.8rem;
        border-radius: 9px;
        min-width: 310px;
    }
    .model-cta a:hover {
        background: #eef6fb;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Fixed model assumptions
# -----------------------------
FIXED_ASSUMPTIONS = {
    "start_timeline": pd.Timestamp("2026-01-01"),
    "length_timeline": 30,
    "con_start": pd.Timestamp("2026-04-01"),
    "con_length": 9,
    "asset_life": 25,
    "inflation_rate": 0.02,
    "tax_rate": 0.25,
    "degradation_rate": 0.003,
    "curtailment": 0.00,
    "price_base": 2026,
    "integrity": 0.01,
    "seasonality_profile": {
        1: 0.035, 2: 0.050, 3: 0.085, 4: 0.105,
        5: 0.125, 6: 0.140, 7: 0.135, 8: 0.115,
        9: 0.090, 10: 0.065, 11: 0.035, 12: 0.020
    }
}


# -----------------------------
# Core model functions
# -----------------------------
def build_model(params):
    timeline = pd.date_range(
        start=params["start_timeline"],
        periods=params["length_timeline"] * 12,
        freq="MS"
    )
    df = pd.DataFrame(index=timeline)

    # Construction
    con_end = params["con_start"] + pd.DateOffset(
        months=params["con_length"] - 1
    )

    df["construction_timeline"] = np.where(
        (df.index >= params["con_start"]) & (df.index <= con_end),
        1,
        0
    )

    df["construction_period"] = (
        df["construction_timeline"].cumsum()
        * df["construction_timeline"]
    )

    progress_ratio = df["construction_period"] / params["con_length"]

    # scipy returns a NumPy array; convert it back to a pandas Series
    # so that it retains the model timeline index and supports .loc.
    raw_s_curve = pd.Series(
        norm.cdf(progress_ratio * 4 - 2),
        index=df.index
    )

    construction_mask = df["construction_timeline"] == 1
    s_min = raw_s_curve.loc[construction_mask].min()
    s_max = raw_s_curve.loc[construction_mask].max()

    df["s_curve"] = np.where(
        construction_mask,
        (raw_s_curve - s_min) / (s_max - s_min),
        0
    )

    total_capex = params["CAPEX_MW"] * params["MW_capacity"]

    df["CAPEX_spending"] = (
        total_capex
        * df["s_curve"].diff().fillna(df["s_curve"])
        * df["construction_timeline"]
    )

    # Operations
    operations_start = con_end + pd.DateOffset(months=1)
    operations_end = (
        operations_start
        + pd.DateOffset(years=params["asset_life"])
        - pd.DateOffset(months=1)
    )

    df["operational_timeline"] = np.where(
        (df.index >= operations_start) & (df.index <= operations_end),
        1,
        0
    )

    df["operational_period"] = (
        df["operational_timeline"].cumsum()
        * df["operational_timeline"]
    )

    df["operational_year"] = (
        ((df["operational_period"] - 1) // 12 + 1)
        * df["operational_timeline"]
    )

    # Electricity production
    df["degradation"] = 1.0
    operational_mask = df["operational_timeline"] == 1

    df.loc[operational_mask, "degradation"] = (
        1
        / (
            (1 + params["degradation_rate"])
            ** (df.loc[operational_mask, "operational_year"] - 1)
        )
    )

    df["seasonality"] = df.index.month.map(
        params["seasonality_profile"]
    )

    df["electricity_exports"] = 0.0
    df.loc[operational_mask, "electricity_exports"] = (
        params["p_yield"]
        * params["MW_capacity"]
        * df.loc[operational_mask, "degradation"]
        * (1 - params["curtailment"])
        * df.loc[operational_mask, "seasonality"]
    )

    # Inflation and OPEX
    df["indexation_period"] = (
        df.index.year - params["price_base"]
    )

    df["inflation"] = (
        (1 + params["inflation_rate"])
        ** df["indexation_period"]
    )

    df["OPEX"] = 0.0
    df.loc[operational_mask, "OPEX"] = (
        params["OPEX_MW"]
        * params["MW_capacity"]
        * df.loc[operational_mask, "inflation"]
        / 12
    )

    df["OPEX_real"] = df["OPEX"] / df["inflation"]

    # Discount rates
    standard_wacc = (
        params["COE"] * (1 - params["debt"])
        + (1 - params["tax_rate"])
        * params["debt"]
        * params["COD"]
    )

    adjusted_wacc_nominal = (
        params["COE"]
        * (1 - params["debt"])
        * (1 + params["tax_rate"])
        + params["debt"] * params["COD"]
    )

    adjusted_wacc_real = (
        (1 + adjusted_wacc_nominal)
        / (1 + params["inflation_rate"])
        - 1
    )

    df["days_from_COD"] = (
        df.index - operations_start
    ).days

    df["standard_discount_factor"] = (
        1
        / (
            (1 + standard_wacc)
            ** (df["days_from_COD"] / 365.25)
        )
    )

    df["adjusted_discount_factor"] = (
        1
        / (
            (1 + adjusted_wacc_real)
            ** (df["days_from_COD"] / 365.25)
        )
    )

    discount_rates = {
        "standard_WACC": standard_wacc,
        "adjusted_WACC_nom": adjusted_wacc_nominal,
        "adjusted_WACC_real": adjusted_wacc_real
    }

    return df, discount_rates


def calculate_standard_lcoe(df):
    total_costs = df["CAPEX_spending"] + df["OPEX"]

    discounted_costs = (
        total_costs * df["standard_discount_factor"]
    )

    discounted_energy = (
        df["electricity_exports"]
        * df["standard_discount_factor"]
    )

    results = {
        "Energy": df["electricity_exports"].sum(),
        "Costs": total_costs.sum(),
        "NPV Energy": discounted_energy.sum(),
        "NPV Costs": discounted_costs.sum()
    }

    results["LCOE"] = (
        results["NPV Costs"]
        / results["NPV Energy"]
        * 1000
    )

    return results


def calculate_adjusted_lcoe(df):
    total_costs = df["CAPEX_spending"] + df["OPEX_real"]

    discounted_costs = (
        total_costs * df["adjusted_discount_factor"]
    )

    discounted_energy = (
        df["electricity_exports"]
        * df["adjusted_discount_factor"]
    )

    results = {
        "Energy": df["electricity_exports"].sum(),
        "Costs": total_costs.sum(),
        "NPV Energy": discounted_energy.sum(),
        "NPV Costs": discounted_costs.sum()
    }

    results["LCOE"] = (
        results["NPV Costs"]
        / results["NPV Energy"]
        * 1000
    )

    return results


def build_sensitivity(params, method):
    rate_steps = [-0.01, -0.005, 0, 0.005, 0.01]
    yield_steps = [-200, -100, 0, 100, 200]

    base_df, base_rates = build_model(params)

    if method == "standard":
        base_rate = base_rates["standard_WACC"]
    else:
        base_rate = base_rates["adjusted_WACC_real"]

    matrix = []

    for rate_delta in rate_steps:
        row = []

        for yield_delta in yield_steps:
            temp_params = params.copy()
            temp_params["p_yield"] = (
                params["p_yield"] + yield_delta
            )

            temp_df, temp_rates = build_model(temp_params)

            if method == "standard":
                sensitised_rate = base_rate + rate_delta

                temp_df["standard_discount_factor"] = (
                    1
                    / (
                        (1 + sensitised_rate)
                        ** (temp_df["days_from_COD"] / 365.25)
                    )
                )

                result = calculate_standard_lcoe(temp_df)

            else:
                sensitised_rate = base_rate + rate_delta

                temp_df["adjusted_discount_factor"] = (
                    1
                    / (
                        (1 + sensitised_rate)
                        ** (temp_df["days_from_COD"] / 365.25)
                    )
                )

                result = calculate_adjusted_lcoe(temp_df)

            row.append(result["LCOE"])

        matrix.append(row)

    row_labels = [
        f"{base_rate + step:.2%}"
        for step in rate_steps
    ]

    column_labels = [
        f"{params['p_yield'] + step:,.0f}"
        for step in yield_steps
    ]

    sensitivity = pd.DataFrame(
        matrix,
        index=row_labels,
        columns=column_labels
    )

    sensitivity.index.name = (
        "WACC \\ Yield"
    )

    return sensitivity


def format_building_blocks(results, currency_symbol):
    return pd.DataFrame(
        {
            "Metric": [
                "Energy",
                "Costs",
                "NPV Energy",
                "NPV Costs",
                "LCOE"
            ],
            "Value": [
                f"{results['Energy']:,.0f} MWh",
                f"{currency_symbol}{results['Costs']:,.0f}k",
                f"{results['NPV Energy']:,.0f} MWh",
                f"{currency_symbol}{results['NPV Costs']:,.0f}k",
                f"{currency_symbol}{results['LCOE']:,.1f}/MWh"
            ]
        }
    )


def style_building_blocks(table):
    return (
        table.style
        .set_properties(**{
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": "14px"
        })
        .set_properties(
            subset=["Metric"],
            **{"font-weight": "600"}
        )
        .set_properties(
            subset=["Value"],
            **{"font-weight": "600", "text-align": "right"}
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("font-family", "Arial, Helvetica, sans-serif"),
                    ("font-weight", "700"),
                    ("background-color", "#1f4e5f"),
                    ("color", "white")
                ]
            }
        ])
    )


def style_sensitivity(table):
    # Highlight the base case at the centre of the 5x5 table.
    styles = pd.DataFrame(
        "",
        index=table.index,
        columns=table.columns
    )
    styles.iloc[2, 2] = (
        "background-color: #dbeef8; "
        "color: #0d3b66; "
        "font-weight: 700; "
        "border: 2px solid #7fb3d5;"
    )

    return (
        table.style
        .format("{:.1f}")
        .apply(lambda _: styles, axis=None)
        .set_properties(**{
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": "13px",
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("font-family", "Arial, Helvetica, sans-serif"),
                    ("font-weight", "700"),
                    ("background-color", "#1f4e5f"),
                    ("color", "white"),
                    ("text-align", "center")
                ]
            }
        ])
    )


# -----------------------------
# App interface
# -----------------------------
st.markdown(
    """
    <section class="ie-hero">
        <div class="ie-brand">Infrastructure Economics</div>
        <h1>Solar LCOE Calculator</h1>
        <p>
            Compare the standard Levelized Cost of Energy methodology with an
            adjusted approach that removes future inflation from the discount
            rate and incorporates the effect of corporation tax.
        </p>
    </section>
    """,
    unsafe_allow_html=True
)

st.subheader("User inputs")

currency_options = {
    "GBP (£)": "£",
    "USD ($)": "$",
    "EUR (€)": "€"
}

input_col_1, input_col_2, input_col_3 = st.columns(3)

with input_col_1:
    currency_name = st.selectbox(
        "Currency",
        options=list(currency_options.keys()),
        index=0
    )
    currency_symbol = currency_options[currency_name]

    plant_capacity = st.number_input(
        "Plant capacity (MW)",
        min_value=1.0,
        value=60.0,
        step=1.0
    )

    electricity_yield = st.number_input(
        "Yield (MWh/MWp)",
        min_value=100.0,
        value=1350.0,
        step=25.0
    )

with input_col_2:
    capex_per_mw = st.number_input(
        f"CAPEX ({currency_symbol}000/MW)",
        min_value=0.0,
        value=650.0,
        step=25.0
    )

    opex_per_mw = st.number_input(
        f"OPEX ({currency_symbol}000/MW/year)",
        min_value=0.0,
        value=25.0,
        step=1.0
    )

    gearing = st.number_input(
        "Gearing ratio (% debt / total funding)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    ) / 100

with input_col_3:
    cost_of_debt = st.number_input(
        "Cost of debt (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.25
    ) / 100

    cost_of_equity = st.number_input(
        "Cost of equity (%)",
        min_value=0.0,
        max_value=40.0,
        value=8.0,
        step=0.25
    ) / 100

params = {
    **FIXED_ASSUMPTIONS,
    "MW_capacity": plant_capacity,
    "p_yield": electricity_yield,
    "CAPEX_MW": capex_per_mw,
    "OPEX_MW": opex_per_mw,
    "COD": cost_of_debt,
    "COE": cost_of_equity,
    "debt": gearing
}

model_df, discount_rates = build_model(params)

standard_results = calculate_standard_lcoe(model_df)
adjusted_results = calculate_adjusted_lcoe(model_df)

standard_sensitivity = build_sensitivity(
    params,
    method="standard"
)

adjusted_sensitivity = build_sensitivity(
    params,
    method="adjusted"
)

st.divider()

comparison_difference = (
    adjusted_results["LCOE"] - standard_results["LCOE"]
)
comparison_percentage = (
    comparison_difference / standard_results["LCOE"]
    if standard_results["LCOE"] != 0
    else 0
)

st.markdown(
    f"""
    <div class="comparison-banner">
        Adjusted LCOE compared with standard methodology<br>
        <strong>{currency_symbol}{comparison_difference:+,.1f}/MWh ({comparison_percentage:+.1%})</strong>
    </div>
    """,
    unsafe_allow_html=True
)

standard_col, adjusted_col = st.columns(2, gap="large")

with standard_col:
    st.header("Standard LCOE")

    st.metric(
        "Calculated LCOE",
        f"{currency_symbol}{standard_results['LCOE']:,.1f}/MWh"
    )

    st.metric(
        "Standard WACC",
        f"{discount_rates['standard_WACC']:.2%}"
    )

    st.subheader("Building blocks")
    st.dataframe(
        style_building_blocks(
            format_building_blocks(standard_results, currency_symbol)
        ),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Sensitivity")
    st.caption(
        "Rows show standard WACC; columns show electricity yield. "
        "The base case is highlighted."
    )
    st.dataframe(
        style_sensitivity(standard_sensitivity),
        use_container_width=True
    )

with adjusted_col:
    st.header("Adjusted LCOE")

    st.metric(
        "Calculated LCOE",
        f"{currency_symbol}{adjusted_results['LCOE']:,.1f}/MWh"
    )

    st.metric(
        "Adjusted real WACC",
        f"{discount_rates['adjusted_WACC_real']:.2%}"
    )

    st.subheader("Building blocks")
    st.dataframe(
        style_building_blocks(
            format_building_blocks(adjusted_results, currency_symbol)
        ),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Sensitivity")
    st.caption(
        "Rows show adjusted real WACC; columns show electricity yield. "
        "The base case is highlighted."
    )
    st.dataframe(
        style_sensitivity(adjusted_sensitivity),
        use_container_width=True
    )

st.divider()

st.subheader("Other model assumptions")

assumption_col_1, assumption_col_2, assumption_col_3 = st.columns(3)

assumption_col_1.metric(
    "Inflation",
    f"{params['inflation_rate']:.1%}"
)

assumption_col_2.metric(
    "Tax rate",
    f"{params['tax_rate']:.1%}"
)

assumption_col_3.metric(
    "Annual degradation",
    f"{params['degradation_rate']:.1%}"
)

assumption_col_4, assumption_col_5, assumption_col_6 = st.columns(3)

assumption_col_4.metric(
    "Curtailment",
    f"{params['curtailment']:.1%}"
)

assumption_col_5.metric(
    "Construction length",
    f"{params['con_length']} months"
)

assumption_col_6.metric(
    "Operational life",
    f"{params['asset_life']} years"
)

st.caption(
    "The calculator also assumes a fixed monthly electricity production profile."
)

st.markdown(
    """
    <div class="model-cta">
        <h3>Run a detailed solar project finance analysis</h3>
        <p>
            The full Excel model allows you to run detailed calculations for a
            solar plant, assess project economics, price PPA contracts, size
            project-finance debt and review the overall financial feasibility of
            a solar investment using a complete set of project assumptions.
        </p>
        <a href="https://infraeconomics.co.uk/solar-plant-project-finance-model/" target="_blank">
            Explore the Solar Project Finance Model
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    """
    <div class="methodology-box">
        <h3>Why adjust the standard LCOE?</h3>
        <p>
            A conventional LCOE calculation normally projects operating costs in
            nominal terms and discounts them using a nominal cost of capital. The
            resulting figure therefore embeds future inflation. When it is
            interpreted as the electricity price required today, it can overstate
            the initial price needed to achieve the target investment return,
            particularly where a PPA or market price is expected to increase with
            inflation.
        </p>
        <p>
            The adjusted calculation removes inflation from both operating costs
            and the discount rate, producing an LCOE expressed in today's money.
            It also adjusts the cost of capital for corporation tax by removing the
            debt tax shield and grossing up the equity return. This provides a more
            representative estimate of the electricity price required to deliver
            the investor's targeted post-tax return.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<h3 class="nav-heading">Continue exploring Infrastructure Economics</h3>', unsafe_allow_html=True)

nav_col_1, nav_col_2, nav_col_3 = st.columns(3)

with nav_col_1:
    st.link_button(
        "Return to the LCOE calculator page",
        "https://infraeconomics.co.uk/lcoe-calculator/",
        use_container_width=True
    )

with nav_col_2:
    st.link_button(
        "Electricity prices and solar investment",
        "https://infraeconomics.co.uk/2026/07/13/are-electricity-prices-high-enough-to-justify-investing-in-a-solar-plant/",
        use_container_width=True
    )

with nav_col_3:
    st.link_button(
        "Read the practical LCOE guide",
        "https://infraeconomics.co.uk/2026/07/09/the-levelized-cost-of-energy-lcoe-a-practical-guide-for-solar-investors/",
        use_container_width=True
    )
