"""
Simulation Engine — Global Supply Chain Operations Control Tower
==================================================================
Runs a baseline (s,S) periodic-review inventory simulation per (Warehouse, SKU),
then re-runs it under a disruption "shock" so the two can be compared.

Shock model implemented here: FACTORY SHUTDOWN.
  - Each factory supplies a real, data-derived share of each SKU's total production
    (multi-sourced network — no single factory is a SKU's sole source).
  - During a shutdown window, that factory's share of the SKU's supply is cut from
    every periodic replenishment order placed in that window.
  - The missing units are backlogged and released as a catch-up order once the
    factory resumes (production doesn't just vanish — it's deferred).

This module is designed to be imported directly by the Streamlit dashboard.
"""
import pandas as pd
import numpy as np

DATA = "data"


def load_data():
    products = pd.read_csv(f"{DATA}/products.csv")
    warehouses = pd.read_csv(f"{DATA}/warehouses.csv")
    factories = pd.read_csv(f"{DATA}/factories.csv")
    sales = pd.read_csv(f"{DATA}/sales_orders.csv", parse_dates=["Date"])
    shipments = pd.read_csv(f"{DATA}/shipments_enriched.csv", parse_dates=["ShipDate", "ArrivalDate"])
    demand_share = pd.read_csv(f"{DATA}/warehouse_demand_share.csv")
    prod_orders = pd.read_csv(f"{DATA}/production_orders.csv", parse_dates=["Date"])
    return products, warehouses, factories, sales, shipments, demand_share, prod_orders


def factory_sku_shares(prod_orders):
    """Each factory's historical share of each SKU's total production volume."""
    vol = prod_orders.groupby(["SKUID", "FactoryID"])["Qty"].sum().reset_index()
    vol["Share"] = vol.groupby("SKUID")["Qty"].transform(lambda x: x / x.sum())
    return vol.pivot(index="SKUID", columns="FactoryID", values="Share").fillna(0)


def build_params(products, warehouses, sales, shipments, demand_share):
    """Precompute mu, sigma, lead time, and order-up-to level S per (Warehouse, SKU)."""
    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    sales_piv = sales.pivot_table(index="Date", columns="SKUID", values="Qty", aggfunc="sum").reindex(all_dates).fillna(0)

    wh_city = dict(zip(warehouses["WarehouseID"], warehouses["Warehouse"]))
    route_lt = shipments.groupby("Destination")["LeadTimeDays"].mean()
    wh_leadtime = {wh: int(round(route_lt.get(city, 12))) for wh, city in wh_city.items()}

    share_map = demand_share.set_index(["SKUID", "WarehouseID"])["DemandShare"]

    R, Z = 7, 1.65
    params = {}
    for sku in products["SKUID"]:
        d_total = sales_piv[sku].values
        for wh in warehouses["WarehouseID"]:
            share = share_map.loc[(sku, wh)]
            d = d_total * share
            mu, sigma = d.mean(), d.std()
            LT = wh_leadtime[wh]
            S = mu * (R + LT) + Z * sigma * np.sqrt(R + LT)
            params[(wh, sku)] = dict(demand=d, mu=mu, sigma=sigma, LT=LT, S=S, R=R)
    return params, all_dates


def simulate(params, all_dates, wh, sku, shock=None, fac_shares=None, demand_shock=None):
    """
    Run the (s,S) simulation for one (warehouse, sku) pair.
    shock: dict(factory_id=str, start_day=int, duration=int) or None - supply-side shock
           (factory shutdown / supplier disruption), applied to replenishment orders.
    fac_shares: DataFrame from factory_sku_shares(), required if shock is given.
    demand_shock: dict(start_day=int, duration=int, multiplier=float) or None - demand-side
           shock. Consumption is multiplied during the window; the (s,S) order-up-to level
           is NOT recalculated (the planning system doesn't "know" about the spike in
           real time), which is what creates the realistic replenishment lag.
    Returns a DataFrame with Date, OnHand, Outbound.
    """
    p = params[(wh, sku)]
    d, S, LT, R = p["demand"], p["S"], p["LT"], p["R"]
    onhand = S
    inv_position = onhand
    pipeline = {}      # arrival_day -> qty
    backlog = 0.0       # units deferred by the shock, waiting for factory to resume

    cut_share = 0.0
    if shock is not None:
        cut_share = fac_shares.loc[sku, shock["factory_id"]] if sku in fac_shares.index and shock["factory_id"] in fac_shares.columns else 0.0

    records = []
    for i, date in enumerate(all_dates):
        arrived = pipeline.pop(i, 0)
        onhand = max(0, onhand + arrived)

        # release backlog once the factory has resumed
        if shock is not None and backlog > 0 and i >= shock["start_day"] + shock["duration"]:
            arrival_day = i + LT
            pipeline[arrival_day] = pipeline.get(arrival_day, 0) + backlog
            inv_position += backlog
            backlog = 0.0

        if i % R == 0:
            order_qty = max(0, S - inv_position)
            if order_qty > 0:
                in_shock_window = shock is not None and shock["start_day"] <= i < shock["start_day"] + shock["duration"]
                if in_shock_window:
                    fulfilled = order_qty * (1 - cut_share)
                    deferred = order_qty * cut_share
                    pipeline[i + LT] = pipeline.get(i + LT, 0) + fulfilled
                    backlog += deferred
                    inv_position += fulfilled   # deferred portion isn't "in position" until released
                else:
                    pipeline[i + LT] = pipeline.get(i + LT, 0) + order_qty
                    inv_position += order_qty

        out = d[i]
        if demand_shock is not None and demand_shock["start_day"] <= i < demand_shock["start_day"] + demand_shock["duration"]:
            out = out * demand_shock["multiplier"]
        onhand = max(0, onhand - out)
        inv_position -= out
        records.append((date, onhand, out))

    return pd.DataFrame(records, columns=["Date", "OnHand", "Outbound"])


def run_demand_spike_scenario(sku, spike_pct, start_date, duration_days, warehouse_ids=None):
    """
    Demand for `sku` increases by spike_pct (e.g. 0.30 = +30%) for duration_days,
    starting start_date. Restrict to warehouse_ids (list) to model a *regional* spike
    (e.g. "demand spike in India" -> warehouse_ids=['WH006']); None = every warehouse.
    The replenishment system does not react in real time to the spike - it only
    catches up on the next scheduled review cycle - which is what produces the gap.
    """
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    params, all_dates = build_params(products, warehouses, sales, shipments, demand_share)
    start_day = (pd.Timestamp(start_date) - all_dates[0]).days
    d_shock = dict(start_day=start_day, duration=duration_days, multiplier=1 + spike_pct)

    whs = warehouse_ids if warehouse_ids else warehouses["WarehouseID"].tolist()
    rows = []
    for wh in whs:
        base = simulate(params, all_dates, wh, sku, demand_shock=None)
        shocked = simulate(params, all_dates, wh, sku, demand_shock=d_shock)
        rows.append(dict(
            SKUID=sku, WarehouseID=wh,
            BaselineStockoutRate=(base["OnHand"] == 0).mean(),
            ShockedStockoutRate=(shocked["OnHand"] == 0).mean(),
            DeltaStockoutRate=(shocked["OnHand"] == 0).mean() - (base["OnHand"] == 0).mean(),
            DaysAtZeroBase=(base["OnHand"] == 0).sum(),
            DaysAtZeroShocked=(shocked["OnHand"] == 0).sum(),
            MinOnHandDuringShock=shocked["OnHand"].min(),
        ))
    return pd.DataFrame(rows)


def run_factory_shutdown_scenario(factory_id, start_date, duration_days):
    """
    Full scenario: baseline vs. shocked, across every (warehouse, SKU) touched by
    this factory. Returns a summary DataFrame + the two full panels.
    """
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    params, all_dates = build_params(products, warehouses, sales, shipments, demand_share)
    fac_shares = factory_sku_shares(prod_orders)
    start_day = (pd.Timestamp(start_date) - all_dates[0]).days
    shock = dict(factory_id=factory_id, start_day=start_day, duration=duration_days)

    affected_skus = fac_shares.columns.tolist() and fac_shares[fac_shares[factory_id] > 0].index.tolist()

    rows = []
    for sku in affected_skus:
        for wh in warehouses["WarehouseID"]:
            base = simulate(params, all_dates, wh, sku, shock=None)
            shocked = simulate(params, all_dates, wh, sku, shock=shock, fac_shares=fac_shares)
            base_stockout = (base["OnHand"] == 0).mean()
            shocked_stockout = (shocked["OnHand"] == 0).mean()
            min_onhand_shocked = shocked["OnHand"].min()
            rows.append(dict(
                SKUID=sku, WarehouseID=wh,
                FactoryShare=fac_shares.loc[sku, factory_id],
                BaselineStockoutRate=base_stockout,
                ShockedStockoutRate=shocked_stockout,
                DeltaStockoutRate=shocked_stockout - base_stockout,
                MinOnHandDuringShock=min_onhand_shocked,
            ))
    summary = pd.DataFrame(rows)
    return summary


def run_supplier_disruption_scenario(component_id, start_date, duration_days):
    """
    A critical component becomes unavailable (supplier bankruptcy / plant fire / export ban).
    Every SKU whose BOM requires this component loses 100% of new production system-wide
    (no substitute assumed) for the duration - regardless of which factory makes it.
    """
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    params, all_dates = build_params(products, warehouses, sales, shipments, demand_share)
    bom = pd.read_csv(f"{DATA}/bom.csv")

    affected_skus = bom[bom["ComponentID"] == component_id]["SKUID"].unique().tolist()
    start_day = (pd.Timestamp(start_date) - all_dates[0]).days
    shock = dict(factory_id="__ALL__", start_day=start_day, duration=duration_days)

    # cut_share=1.0 uniformly (full stop), independent of any single factory's share
    fake_shares = pd.DataFrame(1.0, index=affected_skus, columns=["__ALL__"])

    rows = []
    for sku in affected_skus:
        for wh in warehouses["WarehouseID"]:
            base = simulate(params, all_dates, wh, sku, shock=None)
            shocked = simulate(params, all_dates, wh, sku, shock=shock, fac_shares=fake_shares)
            rows.append(dict(
                SKUID=sku, WarehouseID=wh,
                BaselineStockoutRate=(base["OnHand"] == 0).mean(),
                ShockedStockoutRate=(shocked["OnHand"] == 0).mean(),
                DeltaStockoutRate=(shocked["OnHand"] == 0).mean() - (base["OnHand"] == 0).mean(),
                MinOnHandDuringShock=shocked["OnHand"].min(),
                DaysAtZeroShocked=(shocked["OnHand"] == 0).sum(),
                DaysAtZeroBase=(base["OnHand"] == 0).sum(),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== FACTORY SHUTDOWN: FAC002, 30 days ===")
    fac_summary = run_factory_shutdown_scenario("FAC002", "2025-06-01", 30)
    print(f"avg baseline stockout={fac_summary['BaselineStockoutRate'].mean():.4f}  "
          f"avg shocked stockout={fac_summary['ShockedStockoutRate'].mean():.4f}")

    print("\n=== SUPPLIER DISRUPTION: COMP001, 21 days ===")
    comp_summary = run_supplier_disruption_scenario("COMP001", "2025-06-01", 21)
    print(f"Affected SKUs: {comp_summary['SKUID'].nunique()} across {len(comp_summary)} warehouse-SKU combos")
    print(f"avg baseline stockout={comp_summary['BaselineStockoutRate'].mean():.4f}  "
          f"avg shocked stockout={comp_summary['ShockedStockoutRate'].mean():.4f}")
    print(f"Combos newly pushed into stockout: "
          f"{((comp_summary['BaselineStockoutRate']==0)&(comp_summary['ShockedStockoutRate']>0)).sum()}/{len(comp_summary)}")
    print(comp_summary.sort_values("DeltaStockoutRate", ascending=False).head(10).to_string(index=False))

    print("\n=== DEMAND SPIKE: SKU105 +30%, 14 days, India (WH006) only ===")
    spike_summary = run_demand_spike_scenario("SKU105", 0.30, "2025-04-01", 14, warehouse_ids=["WH006"])
    print(spike_summary.to_string(index=False))

    print("\n=== DEMAND SPIKE: SKU105 +30%, 14 days, ALL warehouses ===")
    spike_all = run_demand_spike_scenario("SKU105", 0.30, "2025-04-01", 14)
    print(f"avg baseline stockout={spike_all['BaselineStockoutRate'].mean():.4f}  "
          f"avg shocked stockout={spike_all['ShockedStockoutRate'].mean():.4f}")
    print(spike_all.sort_values("DeltaStockoutRate", ascending=False).head(8).to_string(index=False))

