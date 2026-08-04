"""
Optimization / Reallocation Engine
=====================================
Reality check before designing this (see simulation_engine.py output): in a
well-run (s,S) inventory system, warehouses rarely hold surplus above their OWN
safety stock - that's the point of the policy. So "steal spare units from
warehouse A to save warehouse B" mostly isn't a real option here (verified: at
the reference planning date in the demand-spike test, every warehouse's on-hand
was AT OR BELOW its own reorder point - zero shippable surplus everywhere).

The real, defensible optimization problem in this data is different and more
useful: when a disruption cuts total available NEW SUPPLY for a SKU (e.g. a
supplier running at reduced capacity), the default/naive behavior is to cut
every warehouse's incoming allocation by the same percentage (pro-rata). That
is NOT necessarily the value-minimizing way to ration scarce supply across a
network with uneven demand and uneven starting buffers. This module solves:

  Given a fixed, reduced total supply pool for a SKU over a disruption window,
  how should it be split across warehouses to minimize total $ value lost to
  stockouts - versus the naive pro-rata default?

This is a genuine constrained-allocation LP (PuLP), not a fantasy "free lateral
transfer" model.
"""
import pandas as pd
import numpy as np
import pulp
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from simulation_engine import load_data, DATA


STOCKOUT_EVENT_PENALTY = 5000  # assumption: fixed cost of ANY stockout event at a warehouse
                                 # (expedite freight, SLA penalty, customer churn) - beyond
                                 # the linear per-unit lost-margin cost. Documented assumption,
                                 # not observed data.


def optimize_supply_allocation(sku, start_date, duration_days, cut_share):
    """
    cut_share: fraction of normal total production capacity lost for this SKU
               during the window (e.g. 0.6 = supplier operating at 40% capacity).

    NOTE ON MODEL DESIGN: an earlier pure-linear version of this LP (minimize
    sum(shortfall * unit_value)) turned out to be mathematically degenerate -
    with one uniform per-unit value and a fixed total supply pool, total $ value
    lost is conserved regardless of how the shortage is split across warehouses.
    Reallocation only has real value once you account for the fact that a full
    stockout event (a warehouse hitting zero) carries costs beyond lost margin -
    expedite freight, SLA penalties, customer churn - that a partial, spread-thin
    shortage does not. This version is a MILP that adds a fixed per-warehouse
    stockout-event penalty, which gives the optimizer a genuine reason to
    concentrate a shortage onto fewer warehouses rather than spread it evenly
    (which is what naive pro-rata cutting does by default).
    """
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    inv = pd.read_csv(f"{DATA}/inventory_daily.csv", parse_dates=["Date"])
    unit_value = float(products.set_index("SKUID").loc[sku, "StdCost"])

    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    start = pd.Timestamp(start_date)
    window_dates = pd.date_range(start, periods=duration_days)

    sales_piv = pd.read_csv(f"{DATA}/sales_orders.csv", parse_dates=["Date"]).pivot_table(
        index="Date", columns="SKUID", values="Qty", aggfunc="sum").reindex(all_dates).fillna(0)
    total_normal_demand_window = sales_piv.loc[sales_piv.index.intersection(window_dates), sku].sum()

    share_map = demand_share.set_index(["SKUID", "WarehouseID"])["DemandShare"]
    whs = warehouses["WarehouseID"].tolist()

    demand_window, onhand_start, need = {}, {}, {}
    for wh in whs:
        share = share_map.loc[(sku, wh)]
        demand_window[wh] = total_normal_demand_window * share
        row = inv[(inv.WarehouseID == wh) & (inv.SKUID == sku) & (inv.Date == start)]
        onhand_start[wh] = float(row["OnHand"].iloc[0]) if len(row) else 0.0
        need[wh] = max(0.0, demand_window[wh] - onhand_start[wh])  # new supply needed to avoid ANY stockout

    total_available_supply = total_normal_demand_window * (1 - cut_share)

    # ---- Naive baseline: everyone's incoming supply cut by the same % (status quo) ----
    naive_alloc = {wh: demand_window[wh] * (1 - cut_share) for wh in whs}
    naive_shortfall = {wh: max(0, need[wh] - naive_alloc[wh]) for wh in whs}
    naive_events = sum(1 for wh in whs if naive_shortfall[wh] > 1e-6)
    naive_cost = sum(naive_shortfall.values()) * unit_value + naive_events * STOCKOUT_EVENT_PENALTY

    # ---- Optimized: MILP reallocates the SAME total supply, minimizing $ + event penalties ----
    prob = pulp.LpProblem("supply_allocation", pulp.LpMinimize)
    alloc = {wh: pulp.LpVariable(f"alloc_{wh}", lowBound=0, upBound=need[wh]) for wh in whs}
    shortfall = {wh: pulp.LpVariable(f"short_{wh}", lowBound=0) for wh in whs}
    z = {wh: pulp.LpVariable(f"z_{wh}", cat="Binary") for wh in whs}
    BIGM = max(need.values()) + 1

    prob += pulp.lpSum(shortfall[wh] * unit_value + z[wh] * STOCKOUT_EVENT_PENALTY for wh in whs)
    prob += pulp.lpSum(alloc[wh] for wh in whs) <= total_available_supply
    for wh in whs:
        prob += shortfall[wh] >= need[wh] - alloc[wh]
        prob += shortfall[wh] <= BIGM * z[wh]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    opt_alloc = {wh: alloc[wh].value() for wh in whs}
    opt_shortfall = {wh: shortfall[wh].value() for wh in whs}
    opt_events = sum(round(z[wh].value()) for wh in whs)
    opt_cost = sum(opt_shortfall.values()) * unit_value + opt_events * STOCKOUT_EVENT_PENALTY

    result = pd.DataFrame({
        "WarehouseID": whs,
        "OnHandAtStart": [round(onhand_start[wh]) for wh in whs],
        "NeedThisWindow": [round(need[wh]) for wh in whs],
        "NaiveAllocation": [round(naive_alloc[wh]) for wh in whs],
        "NaiveShortfall": [round(naive_shortfall[wh]) for wh in whs],
        "NaiveStockoutEvent": [naive_shortfall[wh] > 1e-6 for wh in whs],
        "OptimizedAllocation": [round(opt_alloc[wh]) for wh in whs],
        "OptimizedShortfall": [round(opt_shortfall[wh]) for wh in whs],
        "OptimizedStockoutEvent": [round(z[wh].value()) == 1 for wh in whs],
    })
    summary = dict(
        sku=sku, unit_value=unit_value, total_available_supply=round(total_available_supply),
        naive_events=naive_events, optimized_events=opt_events,
        naive_total_cost=round(naive_cost), optimized_total_cost=round(opt_cost),
        savings=round(naive_cost - opt_cost),
        savings_pct=round((naive_cost - opt_cost) / naive_cost * 100, 1) if naive_cost else 0,
    )
    return result, summary


if __name__ == "__main__":
    result, summary = optimize_supply_allocation("SKU118", "2025-06-01", 21, cut_share=0.6)
    print(result.to_string(index=False))
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
