"""
Global Supply Chain Operations Control Tower — Streamlit Dashboard
=====================================================================
Ties together simulation_engine.py, forecasting_engine.py, and
optimization_engine.py into one interactive app.

Run locally:   streamlit run dashboard.py
Deploy:        push this repo to GitHub, then deploy on share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from simulation_engine import load_data, build_params, simulate, factory_sku_shares
from forecasting_engine import forecast_sku, project_inventory_forward
from optimization_engine import optimize_supply_allocation

st.set_page_config(page_title="Global Ops Control Tower", layout="wide", page_icon="🌐")

# ---- Known coordinates (not in source data - added for map plotting) ----
WH_COORDS = {
    "Singapore": (1.3521, 103.8198), "Dubai": (25.2048, 55.2708),
    "Rotterdam": (51.9244, 4.4777), "Chicago": (41.8781, -87.6298),
    "Sydney": (-33.8688, 151.2093), "Mumbai": (19.0760, 72.8777),
    "Sao Paulo": (-23.5505, -46.6333), "Johannesburg": (-26.2041, 28.0473),
}
FAC_COORDS = {
    "Shenzhen": (22.5431, 114.0579), "Zhengzhou": (34.7466, 113.6254),
    "Ho Chi Minh City": (10.8231, 106.6297), "Chennai": (13.0827, 80.2707),
    "Bangkok": (13.7563, 100.5018), "Noida": (28.5355, 77.3910),
}


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_params():
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = get_data()
    return build_params(products, warehouses, sales, shipments, demand_share)


@st.cache_data
def get_fac_shares():
    _, _, _, _, _, _, prod_orders = get_data()
    return factory_sku_shares(prod_orders)


@st.cache_data
def get_inventory_daily():
    return pd.read_csv("data/inventory_daily.csv", parse_dates=["Date"])


def stockout_color(rate):
    if rate < 0.02:
        return [46, 160, 67]      # green
    elif rate < 0.05:
        return [230, 168, 23]     # amber
    else:
        return [214, 39, 40]      # red


products, warehouses, factories, sales, shipments, demand_share, prod_orders = get_data()
params, all_dates = get_params()
fac_shares = get_fac_shares()
inv_daily = get_inventory_daily()

st.title("🌐 Global Supply Chain Operations Control Tower")
st.caption("Inspired by the operational challenges faced by global consumer electronics manufacturers.")

tab_overview, tab_scenario, tab_forecast = st.tabs(["📍 Network Overview", "⚠️ Disruption Scenarios", "📈 Forecast & Reallocation"])

# ============================================================= OVERVIEW TAB
with tab_overview:
    col_map, col_kpi = st.columns([2, 1])

    sku_for_map = st.selectbox("Show network health for SKU:", products["SKUID"].tolist(), key="overview_sku")

    wh_status = []
    for wh in warehouses["WarehouseID"]:
        combo = inv_daily[(inv_daily.WarehouseID == wh) & (inv_daily.SKUID == sku_for_map)]
        rate = (combo["OnHand"] == 0).mean()
        city = warehouses.set_index("WarehouseID").loc[wh, "Warehouse"]
        lat, lon = WH_COORDS[city]
        wh_status.append(dict(WarehouseID=wh, City=city, lat=lat, lon=lon,
                               StockoutRate=rate, color=stockout_color(rate)))
    wh_df = pd.DataFrame(wh_status)

    fac_rows = []
    for _, row in factories.iterrows():
        lat, lon = FAC_COORDS[row["City"]]
        fac_rows.append(dict(FactoryID=row["FactoryID"], Factory=row["Factory"], City=row["City"],
                              lat=lat, lon=lon, Capacity=row["Capacity"]))
    fac_df = pd.DataFrame(fac_rows)

    with col_map:
        wh_layer = pdk.Layer(
            "ScatterplotLayer", data=wh_df, get_position="[lon, lat]",
            get_fill_color="color", get_radius=180000, pickable=True,
        )
        fac_layer = pdk.Layer(
            "ScatterplotLayer", data=fac_df, get_position="[lon, lat]",
            get_fill_color=[70, 130, 180], get_radius=120000, pickable=True,
        )
        view_state = pdk.ViewState(latitude=15, longitude=40, zoom=1.1)
        st.pydeck_chart(pdk.Deck(
            layers=[wh_layer, fac_layer], initial_view_state=view_state,
            map_style=None,
            tooltip={"text": "{City}\nStockout rate: {StockoutRate}"},
        ))
        st.caption("🔵 Factories   🟢 Healthy warehouse (<2% stockout)   🟡 Watch (2-5%)   🔴 At risk (>5%)")

    with col_kpi:
        st.subheader("Network KPIs")
        avg_stockout = wh_df["StockoutRate"].mean()
        st.metric("Avg. warehouse stockout rate", f"{avg_stockout*100:.2f}%")
        st.metric("Warehouses at risk (>5%)", int((wh_df["StockoutRate"] > 0.05).sum()))
        st.metric("Total factory capacity", f"{factories['Capacity'].sum():,} units/day")
        st.metric("SKUs tracked", len(products))
        st.dataframe(wh_df[["WarehouseID", "City", "StockoutRate"]].style.format({"StockoutRate": "{:.2%}"}),
                     hide_index=True, width='stretch')

# ============================================================= SCENARIO TAB
with tab_scenario:
    st.subheader("Run a disruption scenario")
    scenario_type = st.radio("Scenario type", ["Factory Shutdown", "Supplier / Component Disruption", "Demand Spike"], horizontal=True)

    col1, col2, col3 = st.columns(3)

    if scenario_type == "Factory Shutdown":
        with col1:
            factory_id = st.selectbox("Factory", factories["FactoryID"] + " - " + factories["Factory"] + " (" + factories["City"] + ")")
            factory_id = factory_id.split(" - ")[0]
        with col2:
            start_date = st.date_input("Start date", pd.Timestamp("2025-06-01"))
        with col3:
            duration = st.slider("Duration (days)", 5, 60, 30)

        if st.button("▶ Run Factory Shutdown Scenario", type="primary"):
            with st.spinner("Simulating..."):
                affected_skus = fac_shares[fac_shares[factory_id] > 0].index.tolist()
                rows = []
                for sku in affected_skus:
                    for wh in warehouses["WarehouseID"]:
                        shock = dict(factory_id=factory_id, start_day=(pd.Timestamp(start_date) - all_dates[0]).days, duration=duration)
                        base = simulate(params, all_dates, wh, sku, shock=None)
                        shocked = simulate(params, all_dates, wh, sku, shock=shock, fac_shares=fac_shares)
                        rows.append(dict(SKUID=sku, WarehouseID=wh,
                                          BaselineStockout=(base.OnHand == 0).mean(),
                                          ShockedStockout=(shocked.OnHand == 0).mean()))
                result = pd.DataFrame(rows)
            st.success(f"{factory_id} supplies {len(affected_skus)} SKUs — simulated across all warehouses.")
            c1, c2 = st.columns(2)
            c1.metric("Avg. baseline stockout rate", f"{result.BaselineStockout.mean()*100:.2f}%")
            c2.metric("Avg. shocked stockout rate", f"{result.ShockedStockout.mean()*100:.2f}%",
                       delta=f"{(result.ShockedStockout.mean()-result.BaselineStockout.mean())*100:+.2f} pp")
            st.dataframe(result.sort_values("ShockedStockout", ascending=False).head(15), hide_index=True, width='stretch')
            st.info("With 6 multi-sourced factories, a single factory's disruption is usually well absorbed by safety stock — this scenario tends to show a small delta. That's a real resilience finding, not a bug.")

    elif scenario_type == "Supplier / Component Disruption":
        with col1:
            sku_choice = st.selectbox("SKU affected", products["SKUID"].tolist())
        with col2:
            start_date = st.date_input("Start date", pd.Timestamp("2025-06-01"), key="supp_start")
        with col3:
            duration = st.slider("Duration (days)", 5, 45, 21, key="supp_dur")
        cut_share = st.slider("Supplier capacity lost (%)", 10, 100, 60, key="supp_cut") / 100

        if st.button("▶ Run Supplier Disruption + Reallocation", type="primary"):
            with st.spinner("Simulating and optimizing..."):
                result, summary = optimize_supply_allocation(sku_choice, str(start_date), duration, cut_share)
            st.success(f"Modeled a {cut_share*100:.0f}% capacity cut on {sku_choice} for {duration} days.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Naive (pro-rata) cost", f"${summary['naive_total_cost']:,}", help=f"{summary['naive_events']} warehouses hit stockout")
            c2.metric("Optimized reallocation cost", f"${summary['optimized_total_cost']:,}", help=f"{summary['optimized_events']} warehouses hit stockout")
            c3.metric("Savings", f"${summary['savings']:,}", delta=f"{summary['savings_pct']:.1f}%")
            st.dataframe(result, hide_index=True, width='stretch')
            st.caption("Naive = every warehouse's incoming supply cut by the same %. Optimized = MILP concentrates the "
                       "unavoidable shortage on fewer warehouses (protecting the rest fully) to minimize stockout-event penalties + lost-margin cost.")

    else:  # Demand Spike
        with col1:
            sku_choice = st.selectbox("SKU", products["SKUID"].tolist(), key="spike_sku")
        with col2:
            wh_choice = st.multiselect("Warehouse(s) affected (blank = all)", warehouses["WarehouseID"].tolist())
        with col3:
            spike_pct = st.slider("Demand increase (%)", 10, 300, 100) / 100
        col4, col5 = st.columns(2)
        with col4:
            start_date = st.date_input("Start date", pd.Timestamp("2025-04-01"), key="spike_start")
        with col5:
            duration = st.slider("Duration (days)", 7, 60, 30, key="spike_dur")

        if st.button("▶ Run Demand Spike Scenario", type="primary"):
            whs = wh_choice if wh_choice else warehouses["WarehouseID"].tolist()
            start_day = (pd.Timestamp(start_date) - all_dates[0]).days
            d_shock = dict(start_day=start_day, duration=duration, multiplier=1 + spike_pct)
            rows, traces = [], {}
            for wh in whs:
                base = simulate(params, all_dates, wh, sku_choice, demand_shock=None)
                shocked = simulate(params, all_dates, wh, sku_choice, demand_shock=d_shock)
                rows.append(dict(WarehouseID=wh, BaselineStockout=(base.OnHand==0).mean(), ShockedStockout=(shocked.OnHand==0).mean()))
                traces[wh] = (base, shocked)
            result = pd.DataFrame(rows)
            st.success(f"+{spike_pct*100:.0f}% demand spike on {sku_choice} for {duration} days.")
            c1, c2 = st.columns(2)
            c1.metric("Avg. baseline stockout rate", f"{result.BaselineStockout.mean()*100:.2f}%")
            c2.metric("Avg. shocked stockout rate", f"{result.ShockedStockout.mean()*100:.2f}%",
                       delta=f"{(result.ShockedStockout.mean()-result.BaselineStockout.mean())*100:+.2f} pp")

            focus_wh = whs[0]
            base, shocked = traces[focus_wh]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base.Date, y=base.OnHand, name="Baseline", line=dict(color="#2ca02c")))
            fig.add_trace(go.Scatter(x=shocked.Date, y=shocked.OnHand, name="With spike", line=dict(color="#d62728")))
            fig.add_vrect(x0=str(start_date), x1=str(pd.Timestamp(start_date)+pd.Timedelta(days=duration)),
                          fillcolor="orange", opacity=0.15, line_width=0)
            fig.update_layout(title=f"On-hand inventory — {sku_choice} @ {focus_wh}", height=400)
            st.plotly_chart(fig, width='stretch')
            st.dataframe(result, hide_index=True, width='stretch')

# ============================================================= FORECAST TAB
with tab_forecast:
    st.subheader("Demand forecast & projected inventory")
    col1, col2, col3 = st.columns(3)
    sku_f = col1.selectbox("SKU", products["SKUID"].tolist(), key="fc_sku")
    wh_f = col2.selectbox("Warehouse", warehouses["WarehouseID"].tolist(), key="fc_wh")
    horizon = col3.slider("Forecast horizon (days)", 30, 180, 90)

    with st.spinner("Forecasting..."):
        fc = forecast_sku(sku_f, horizon)
        proj = project_inventory_forward(sku_f, wh_f, horizon)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fc.Date, y=fc.ForecastQty, name="Forecast", line=dict(color="#1f77b4")))
    fig1.add_trace(go.Scatter(x=fc.Date, y=fc.Upper80, name="80% upper", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=fc.Date, y=fc.Lower80, name="80% interval", fill="tonexty",
                               line=dict(width=0), fillcolor="rgba(31,119,180,0.2)"))
    fig1.update_layout(title=f"Demand forecast — {sku_f} (national)", height=350)
    st.plotly_chart(fig1, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=proj.Date, y=proj.ProjectedOnHand, name="Projected on-hand", line=dict(color="#2ca02c")))
    stockout_days = proj[proj.PredictedStockout]
    if len(stockout_days):
        fig2.add_trace(go.Scatter(x=stockout_days.Date, y=stockout_days.ProjectedOnHand, mode="markers",
                                   name="Predicted stockout", marker=dict(color="red", size=8)))
    fig2.update_layout(title=f"Projected inventory — {sku_f} @ {wh_f}", height=350)
    st.plotly_chart(fig2, width='stretch')

    if len(stockout_days):
        st.warning(f"⚠️ Predicted stockout: {len(stockout_days)} day(s) in the next {horizon} days, "
                    f"first on **{stockout_days.iloc[0]['Date'].date()}**.")
    else:
        st.success(f"✅ No stockout predicted for {sku_f} @ {wh_f} in the next {horizon} days.")

    st.caption("Forecast uses Holt's linear trend exponential smoothing (no weekly seasonality found in the data). "
               "The dataset's own included 'Forecast' column was found to have unrealistically low error (~5% MAPE) "
               "for such noisy demand — consistent with leaking same-day actuals rather than being a genuine "
               "out-of-sample forecast, so it isn't used here. See data/README_data_prep.md for details.")
