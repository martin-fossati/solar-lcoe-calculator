import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(
    page_title="Solar LCOE Calculator",
    page_icon="☀️",
    layout="wide"
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
    raw_s_curve = norm.cdf(progress_ratio * 4 - 2)

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


def format_building_blocks(results):
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
                f"£{results['Costs']:,.0f}k",
                f"{results['NPV Energy']:,.0f} MWh",
                f"£{results['NPV Costs']:,.0f}k",
                f"£{results['LCOE']:,.1f}/MWh"
            ]
        }
    )


# -----------------------------
# App interface
# -----------------------------
st.title("Solar LCOE Calculator")

st.write(
    "Compare the standard LCOE methodology with an adjusted approach "
    "that removes future inflation from the discount rate and includes "
    "the tax impact in the cost of capital."
)

st.subheader("User inputs")

input_col_1, input_col_2, input_col_3 = st.columns(3)

with input_col_1:
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

    capex_per_mw = st.number_input(
        "CAPEX (£000/MW)",
        min_value=0.0,
        value=650.0,
        step=25.0
    )

with input_col_2:
    opex_per_mw = st.number_input(
        "OPEX (£000/MW/year)",
        min_value=0.0,
        value=25.0,
        step=1.0
    )

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

with input_col_3:
    gearing = st.number_input(
        "Gearing ratio (% debt / total funding)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    ) / 100

    st.info(
        "Inflation, tax, degradation and curtailment are fixed assumptions "
        "in this online calculator."
    )

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

standard_col, adjusted_col = st.columns(2)

with standard_col:
    st.header("Standard LCOE")

    st.metric(
        "Calculated LCOE",
        f"£{standard_results['LCOE']:,.1f}/MWh"
    )

    st.caption(
        f"Standard WACC: "
        f"{discount_rates['standard_WACC']:.2%}"
    )

    st.subheader("Building blocks")

    st.dataframe(
        format_building_blocks(standard_results),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Sensitivity")

    st.caption(
        "Rows show the standard WACC; columns show electricity yield."
    )

    st.dataframe(
        standard_sensitivity.style.format("{:.1f}"),
        use_container_width=True
    )

with adjusted_col:
    st.header("Adjusted LCOE")

    st.metric(
        "Calculated LCOE",
        f"£{adjusted_results['LCOE']:,.1f}/MWh",
        delta=(
            f"{adjusted_results['LCOE'] - standard_results['LCOE']:+.1f} "
            "vs standard"
        )
    )

    st.caption(
        f"Adjusted nominal WACC: "
        f"{discount_rates['adjusted_WACC_nom']:.2%} | "
        f"Adjusted real WACC: "
        f"{discount_rates['adjusted_WACC_real']:.2%}"
    )

    st.subheader("Building blocks")

    st.dataframe(
        format_building_blocks(adjusted_results),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Sensitivity")

    st.caption(
        "Rows show the adjusted real WACC; columns show electricity yield."
    )

    st.dataframe(
        adjusted_sensitivity.style.format("{:.1f}"),
        use_container_width=True
    )

st.divider()

st.subheader("Other model assumptions")

assumption_col_1, assumption_col_2, assumption_col_3, assumption_col_4 = (
    st.columns(4)
)

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

assumption_col_4.metric(
    "Curtailment",
    f"{params['curtailment']:.1%}"
)

st.caption(
    "The calculator also assumes a 25-year operating life, a 9-month "
    "construction period and a fixed monthly production profile."
)

st.info(
    "Need to change inflation, tax, degradation, curtailment, construction "
    "timing or other project-specific assumptions? The full Excel model "
    "provides access to the complete assumption set and detailed calculations."
)

# Replace the placeholder below with the actual product page.
st.link_button(
    "View the full Excel model",
    "https://infraeconomics.co.uk/calculator-tools/"
)