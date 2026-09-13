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

# Light-touch visual polish - keep native Streamlit theming/colors so this still
# adapts to light/dark mode, just tighten spacing and de-emphasize chrome a bit.
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
[data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.85; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

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

# The one warehouse deliberately given a leaner stocking policy (see the
# "warehouse_demand_share.csv v2 + warehouses.csv TargetZ/BufferMultiplier"
# note in data/README_data_prep.md for the full data-generation writeup).
# Pulling the reason from there rather than restating it differently here.
WH004_REASON = (
    "Chicago (WH004) is deliberately run leaner than the rest of the network — "
    "it has the lowest total order volume of the 8 warehouses, so it's the one "
    "place a real network would plausibly under-invest in safety stock. That "
    "makes it more exposed during disruptions, so seeing it flagged more often "
    "than the others here is expected, not a bug."
)


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
N_FACTORIES = len(factories)

st.title("🌐 Global Supply Chain Operations Control Tower")
st.caption("Inspired by the operational challenges faced by global consumer electronics manufacturers.")

st.markdown(
    "This tool simulates a global supply-chain network — **8 warehouses**, "
    f"**{N_FACTORIES} factories**, **{len(products)} products** — and stress-tests it "
    "against real disruptions: a factory going down, a key supplier failing, or demand "
    "for a product suddenly spiking. Use it to see which parts of the network are "
    "exposed, how big the impact actually is, and how forecasting demand ahead of time "
    "can help a warehouse avoid running out of stock. As a rule of thumb across this "
    "app: 🟢 **green** means healthy, 🟡 **amber** means worth watching, 🔴 **red** means "
    "genuinely at risk."
)

tab_overview, tab_scenario, tab_forecast = st.tabs(["📍 Network Overview", "⚠️ Disruption Scenarios", "📈 Forecast & Reallocation"])

# ============================================================= OVERVIEW TAB
with tab_overview:
    st.caption("A live health check of every warehouse in the network — green means healthy, red means at risk of running out of stock.")

    with st.container(border=True):
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

    col_map, col_kpi = st.columns([2, 1])

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
        st.caption(
            "🔵 Factories (where products are made)   🟢 Healthy warehouse (<2% stockout)   "
            "🟡 Watch (2-5%)   🔴 At risk (>5%) — a warehouse's **stockout rate** is simply "
            "the share of days it has zero units of this product on the shelf."
        )
        wh004_rate = wh_df.set_index("WarehouseID").loc["WH004", "StockoutRate"] if "WH004" in wh_df["WarehouseID"].values else 0
        if wh004_rate > 0.05:
            st.info(f"🔴 **Why is Chicago (WH004) red?** {WH004_REASON}")

    with col_kpi:
        st.subheader("Network KPIs")
        avg_stockout = wh_df["StockoutRate"].mean()
        st.metric(
            "Avg. warehouse stockout rate", f"{avg_stockout*100:.2f}%",
            help="The share of days a warehouse has zero units of this product on the "
                 "shelf, averaged across all 8 warehouses — lower is better. Under 2% is healthy.",
        )
        st.metric(
            "Warehouses at risk (>5%)", int((wh_df["StockoutRate"] > 0.05).sum()),
            help="Number of warehouses currently out of stock on this product more than "
                 "1 day in 20 — i.e. flagged red on the map.",
        )
        st.metric(
            "Total factory capacity", f"{factories['Capacity'].sum():,} units/day",
            help="Combined units per day all 6 factories in the network can produce, across all products.",
        )
        st.metric(
            "SKUs tracked", len(products),
            help="Number of distinct products (SKUs - Stock Keeping Units, i.e. individual product codes) tracked in this simulation.",
        )
        st.dataframe(wh_df[["WarehouseID", "City", "StockoutRate"]].style.format({"StockoutRate": "{:.2%}"}),
                     hide_index=True, width='stretch')

# ============================================================= SCENARIO TAB
with tab_scenario:
    st.caption("Simulate what happens when something goes wrong - a factory shuts down, a key "
               "supplier fails, or demand suddenly spikes - and see how well the network holds up.")
    scenario_type = st.radio("Scenario type", ["Factory Shutdown", "Supplier / Component Disruption", "Demand Spike"], horizontal=True)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        if scenario_type == "Factory Shutdown":
            st.caption("A factory goes offline for a period of time. See how much stockout risk that creates "
                       "for the products it makes, given the other factories that can (partly) cover for it.")
            with col1:
                factory_id = st.selectbox("Factory", factories["FactoryID"] + " - " + factories["Factory"] + " (" + factories["City"] + ")")
                factory_id = factory_id.split(" - ")[0]
            with col2:
                start_date = st.date_input("Start date", pd.Timestamp("2025-06-01"))
            with col3:
                duration = st.slider("Duration (days)", 5, 60, 30)
            run_clicked = st.button("▶ Run Factory Shutdown Scenario", type="primary")

        elif scenario_type == "Supplier / Component Disruption":
            st.caption("A key supplier can only deliver a fraction of the components a product needs. "
                       "Compare cutting every warehouse's supply by the same amount (the naive default) "
                       "against an optimizer that reallocates the limited supply more intelligently.")
            with col1:
                sku_choice = st.selectbox("SKU affected", products["SKUID"].tolist())
            with col2:
                start_date = st.date_input("Start date", pd.Timestamp("2025-06-01"), key="supp_start")
            with col3:
                duration = st.slider("Duration (days)", 5, 45, 21, key="supp_dur")
            cut_share = st.slider(
                "Supplier capacity lost (%)", 10, 100, 60, key="supp_cut",
                help="How much of the supplier's normal output is unavailable during the disruption window.",
            ) / 100
            run_clicked = st.button("▶ Run Supplier Disruption + Reallocation", type="primary")

        else:  # Demand Spike
            st.caption("Demand for a product suddenly jumps - e.g. a viral trend or a regional surge. "
                       "See how long it takes warehouses to run dry before the network catches up.")
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
            run_clicked = st.button("▶ Run Demand Spike Scenario", type="primary")

    # --------------------------------------------------------- RESULTS
    if scenario_type == "Factory Shutdown" and run_clicked:
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
        with st.container(border=True):
            delta_pp = (result.ShockedStockout.mean() - result.BaselineStockout.mean()) * 100
            st.success(f"{factory_id} supplies {len(affected_skus)} products — simulated across all 8 warehouses.")
            if delta_pp < 0.5:
                st.markdown(
                    f"**What this means:** this shutdown barely moved the needle "
                    f"(+{delta_pp:.2f} percentage points of stockout risk) because "
                    f"{N_FACTORIES - 1} other factories can pick up the slack. That's a real "
                    "resilience finding about this network, not a bug in the simulation."
                )
            else:
                st.markdown(
                    f"**What this means:** this shutdown raised the average stockout rate by "
                    f"**{delta_pp:.2f} percentage points** — a meaningful hit, because the products "
                    f"{factory_id} makes aren't as well covered by the other {N_FACTORIES - 1} factories."
                )
            c1, c2 = st.columns(2)
            c1.metric("Avg. stockout rate — normal conditions", f"{result.BaselineStockout.mean()*100:.2f}%",
                      help="What these products' stockout rate looks like with no disruption.")
            c2.metric("Avg. stockout rate — during shutdown", f"{result.ShockedStockout.mean()*100:.2f}%",
                       delta=f"{delta_pp:+.2f} pp",
                       help="What it becomes while this factory is offline. 'pp' = percentage points of change.")
            st.caption("Worst-affected product/warehouse combinations (highest stockout rate during the shutdown):")
            st.dataframe(result.sort_values("ShockedStockout", ascending=False).head(15), hide_index=True, width='stretch')

    elif scenario_type == "Supplier / Component Disruption" and run_clicked:
        with st.spinner("Simulating and optimizing..."):
            result, summary = optimize_supply_allocation(sku_choice, str(start_date), duration, cut_share)
        with st.container(border=True):
            st.success(f"Modeled a {cut_share*100:.0f}% capacity cut on {sku_choice} for {duration} days.")
            st.markdown(
                f"**What this means:** simply cutting every warehouse's supply by the same "
                f"percentage (the naive default) triggers stockouts at **{summary['naive_events']} "
                f"warehouses**. Reallocating that same limited supply more deliberately - giving "
                f"more to the warehouses that need it most instead of spreading it thin everywhere - "
                f"cuts that down to **{summary['optimized_events']} warehouses** and saves an "
                f"estimated **${summary['savings']:,}** ({summary['savings_pct']:.1f}%) in lost sales and stockout costs."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Cost of naive approach", f"${summary['naive_total_cost']:,}",
                      help=f"'Naive' = every warehouse's incoming supply is cut by the same percentage, "
                           f"regardless of need. {summary['naive_events']} warehouses run out of stock this way.")
            c2.metric("Cost of optimized approach", f"${summary['optimized_total_cost']:,}",
                      help=f"'Optimized' = a solver decides how to split the limited supply across warehouses "
                           f"to minimize total cost, instead of splitting it evenly. {summary['optimized_events']} warehouses run out this way.")
            c3.metric("Savings", f"${summary['savings']:,}", delta=f"{summary['savings_pct']:.1f}%",
                      help="Estimated $ saved by reallocating supply intelligently instead of cutting everyone by the same percentage.")
            st.dataframe(result, hide_index=True, width='stretch')
            st.caption(
                "**Naive** = every warehouse's incoming supply cut by the same %, regardless of actual need. "
                "**Optimized** = a solver (mathematically, a MILP - Mixed-Integer Linear Program, a method for "
                "finding the best way to divide a limited resource) concentrates the unavoidable shortage on "
                "fewer warehouses, protecting the rest fully, to minimize lost-sales cost plus a fixed penalty "
                "per warehouse that fully runs out."
            )

    elif scenario_type == "Demand Spike" and run_clicked:
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
        with st.container(border=True):
            delta_pp = (result.ShockedStockout.mean() - result.BaselineStockout.mean()) * 100
            st.success(f"+{spike_pct*100:.0f}% demand spike on {sku_choice} for {duration} days.")
            if delta_pp < 0.5:
                st.markdown(
                    f"**What this means:** the safety stock these warehouses already carry absorbed "
                    f"almost all of this spike (+{delta_pp:.2f} percentage points of stockout risk) - "
                    "a genuinely mild spike relative to their buffer, not a broken scenario."
                )
            else:
                st.markdown(
                    f"**What this means:** this spike raised the average stockout rate by "
                    f"**{delta_pp:.2f} percentage points**, because the warehouse's replenishment "
                    "system only catches up on its next scheduled review cycle - it doesn't react "
                    "to the spike in real time."
                )
                if "WH004" in whs:
                    st.caption(f"ℹ️ {WH004_REASON}")
            c1, c2 = st.columns(2)
            c1.metric("Avg. stockout rate — normal conditions", f"{result.BaselineStockout.mean()*100:.2f}%",
                      help="What this product's stockout rate looks like with no demand spike.")
            c2.metric("Avg. stockout rate — during spike", f"{result.ShockedStockout.mean()*100:.2f}%",
                       delta=f"{delta_pp:+.2f} pp",
                       help="What it becomes during the spike. 'pp' = percentage points of change.")

            focus_wh = whs[0]
            base, shocked = traces[focus_wh]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base.Date, y=base.OnHand, name="Baseline", line=dict(color="#2ca02c")))
            fig.add_trace(go.Scatter(x=shocked.Date, y=shocked.OnHand, name="With spike", line=dict(color="#d62728")))
            fig.add_vrect(x0=str(start_date), x1=str(pd.Timestamp(start_date)+pd.Timedelta(days=duration)),
                          fillcolor="orange", opacity=0.15, line_width=0)
            fig.update_layout(title=f"On-hand inventory (units on the shelf) — {sku_choice} @ {focus_wh}", height=400)
            st.plotly_chart(fig, width='stretch')
            st.caption("The orange band marks the spike window. Where the green/red line touches zero, that warehouse is out of stock.")
            st.dataframe(result, hide_index=True, width='stretch')

# ============================================================= FORECAST TAB
with tab_forecast:
    st.caption("Predict future demand from historical sales, then project whether a warehouse will run "
               "out of stock before its next resupply arrives - and compare smarter supply reallocation "
               "against the naive default.")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        sku_f = col1.selectbox("SKU", products["SKUID"].tolist(), key="fc_sku")
        wh_f = col2.selectbox("Warehouse", warehouses["WarehouseID"].tolist(), key="fc_wh")
        horizon = col3.slider(
            "Forecast horizon (days)", 30, 180, 90,
            help="How many days into the future to predict demand and project inventory.",
        )
        if wh_f == "WH004":
            st.caption(f"ℹ️ {WH004_REASON}")

    with st.spinner("Forecasting..."):
        fc = forecast_sku(sku_f, horizon)
        proj = project_inventory_forward(sku_f, wh_f, horizon)

    hist = sales[sales.SKUID == sku_f].set_index("Date")["Qty"].sort_index()
    hist_window = hist[hist.index >= fc.Date.min() - pd.Timedelta(days=90)]

    with st.container(border=True):
        st.subheader("Demand forecast")
        st.caption("The grey line is what actually happened in the last 90 days. The blue line and shaded "
                   "band are the prediction for what comes next, with an 80% confidence range (we expect "
                   "the real number to land inside that band 4 times out of 5).")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=hist_window.index, y=hist_window.values, name="Actual (last 90d)",
                                   line=dict(color="#7f7f7f", width=1), opacity=0.5))
        fig1.add_trace(go.Scatter(x=fc.Date, y=fc.ForecastQty, name="Forecast", line=dict(color="#1f77b4")))
        fig1.add_trace(go.Scatter(x=fc.Date, y=fc.Upper80, name="80% upper", line=dict(width=0), showlegend=False))
        fig1.add_trace(go.Scatter(x=fc.Date, y=fc.Lower80, name="80% interval", fill="tonexty",
                                   line=dict(width=0), fillcolor="rgba(31,119,180,0.2)"))
        fig1.update_layout(title=f"Demand forecast — {sku_f} (national)", height=350)
        st.plotly_chart(fig1, width='stretch')
        st.caption(
            "Forecast method: Holt's linear trend exponential smoothing (no weekly seasonality was found "
            "in the data, so the model doesn't invent a pattern that isn't really there). Accuracy is "
            "measured with **MAPE** (Mean Absolute Percentage Error - on average, how far off a prediction "
            "is, as a % of the real number; lower is better). This model runs ~44% MAPE on held-out data. "
            "The dataset's own included 'Forecast' column scores an unrealistically low ~5% MAPE for such "
            "noisy demand - a sign it likely leaked same-day actuals rather than being a genuine advance "
            "prediction, so it isn't used here. See data/README_data_prep.md for the full writeup."
        )

    with st.container(border=True):
        st.subheader("Will this warehouse run out of stock?")
        st.caption(
            "Projects the forecasted demand above forward through this warehouse's restocking routine "
            "(it reorders on a fixed schedule and waits out a shipping lead time before new stock arrives) "
            "to flag the specific days it's predicted to hit zero units on the shelf."
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=proj.Date, y=proj.ProjectedOnHand, name="Projected on-hand", line=dict(color="#2ca02c")))
        stockout_days = proj[proj.PredictedStockout]
        if len(stockout_days):
            fig2.add_trace(go.Scatter(x=stockout_days.Date, y=stockout_days.ProjectedOnHand, mode="markers",
                                       name="Predicted stockout", marker=dict(color="red", size=8)))
        fig2.update_layout(title=f"Projected inventory — {sku_f} @ {wh_f}", height=350)
        st.plotly_chart(fig2, width='stretch')

        if len(stockout_days):
            st.warning(f"⚠️ **What this means:** {sku_f} at {wh_f} is predicted to run out of stock on "
                       f"**{len(stockout_days)} day(s)** in the next {horizon} days, starting "
                       f"**{stockout_days.iloc[0]['Date'].date()}** - worth reordering earlier or "
                       "requesting extra supply ahead of that date.")
        else:
            st.success(f"✅ **What this means:** {sku_f} at {wh_f} is not predicted to run out of stock "
                       f"in the next {horizon} days under normal conditions.")
