"""
Forecasting Engine — Global Supply Chain Operations Control Tower
====================================================================
Data check before modeling: SKU-level daily demand shows no weekly seasonality
(autocorrelation ~0 at lag 7) but a real, statistically significant upward trend
across all 30 SKUs (linregress p<0.05 for every SKU). So: Holt's linear trend
exponential smoothing (trend, no seasonality) - not naive persistence, not a
seasonal model that would be fitting noise.

Provides:
  - backtest_all_skus()      -> MAPE of our model vs. naive baseline vs. the
                                 dataset's own given Forecast column, on a common
                                 held-out window (fair comparison)
  - forecast_sku(sku, h)     -> forward point forecast + 80% interval, h days
                                 beyond the last observed date
  - project_inventory_forward -> feeds the forecast into the (s,S) simulation
                                 engine to answer "which warehouse stocks out
                                 in the next N days", starting from the last
                                 actual on-hand position
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from simulation_engine import load_data, build_params, DATA

TRAIN_END = "2025-10-27"   # last 65 days held out for backtesting


def _series(sales, sku):
    piv = sales.pivot_table(index="Date", columns="SKUID", values="Qty", aggfunc="sum")
    return piv[sku].asfreq("D")


def fit_holt(y):
    model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None)
    return model.fit(optimized=True)


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs(actual - pred) / actual) * 100


def backtest_all_skus():
    """Compare our Holt model vs naive (last value) vs the dataset's own ForecastQty,
    all evaluated on the same held-out 65-day window, per SKU."""
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    forecast_given = pd.read_csv(f"{DATA}/forecast.csv", parse_dates=["Date"])

    rows = []
    for sku in products["SKUID"]:
        y = _series(sales, sku)
        train, test = y[:TRAIN_END], y[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):]
        fit = fit_holt(train)
        our_pred = fit.forecast(len(test))
        naive_pred = np.repeat(train.iloc[-1], len(test))

        given = forecast_given[forecast_given["SKUID"] == sku].set_index("Date")
        given_test = given.loc[test.index, "ForecastQty"]

        rows.append(dict(
            SKUID=sku,
            MAPE_ours=mape(test.values, our_pred.values),
            MAPE_naive=mape(test.values, naive_pred),
            MAPE_given=mape(test.values, given_test.values),
        ))
    return pd.DataFrame(rows)


def forecast_sku(sku, horizon=90):
    """Forward forecast, horizon days beyond the last observed date (2025-12-31)."""
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    y = _series(sales, sku)
    fit = fit_holt(y)
    pred = fit.forecast(horizon)
    resid_std = np.std(fit.fittedvalues - y)
    future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon)
    out = pd.DataFrame({
        "Date": future_dates,
        "ForecastQty": pred.values,
        "Lower80": np.maximum(0, pred.values - 1.28 * resid_std),
        "Upper80": pred.values + 1.28 * resid_std,
    })
    return out


def project_inventory_forward(sku, wh, horizon=90):
    """
    Starting from the last actual on-hand position (Dec 31 2025), project inventory
    forward using the FORECASTED demand (not historical) under the same (s,S) policy,
    and flag which future dates breach zero - i.e. predicted stockouts.
    """
    products, warehouses, factories, sales, shipments, demand_share, prod_orders = load_data()
    params, all_dates = build_params(products, warehouses, sales, shipments, demand_share)
    inv_hist = pd.read_csv(f"{DATA}/inventory_daily.csv", parse_dates=["Date"])
    last_actual = inv_hist[(inv_hist.WarehouseID == wh) & (inv_hist.SKUID == sku) & (inv_hist.Date == inv_hist.Date.max())]
    onhand = float(last_actual["OnHand"].iloc[0])

    p = params[(wh, sku)]
    S, LT, R = p["S"], p["LT"], p["R"]
    share = demand_share.set_index(["SKUID", "WarehouseID"])["DemandShare"].loc[(sku, wh)]

    fc = forecast_sku(sku, horizon)
    fc["WarehouseDemand"] = fc["ForecastQty"] * share

    inv_position = onhand
    pipeline = {}
    records = []
    for i, row in fc.iterrows():
        arrived = pipeline.pop(i, 0)
        onhand = max(0, onhand + arrived)
        if i % R == 0:
            order_qty = max(0, S - inv_position)
            if order_qty > 0:
                pipeline[i + LT] = pipeline.get(i + LT, 0) + order_qty
                inv_position += order_qty
        out = row["WarehouseDemand"]
        onhand = max(0, onhand - out)
        inv_position -= out
        records.append((row["Date"], onhand, out))
    proj = pd.DataFrame(records, columns=["Date", "ProjectedOnHand", "ProjectedOutbound"])
    proj["PredictedStockout"] = proj["ProjectedOnHand"] == 0
    return proj


if __name__ == "__main__":
    print("=== BACKTEST: our model vs naive vs dataset's given forecast (65-day holdout) ===")
    bt = backtest_all_skus()
    print(bt.describe())
    print(f"\nOur model wins vs naive on {(bt.MAPE_ours < bt.MAPE_naive).sum()}/30 SKUs")
    print(f"Our model wins vs given forecast on {(bt.MAPE_ours < bt.MAPE_given).sum()}/30 SKUs")

    print("\n=== FORWARD FORECAST: SKU105, next 14 days ===")
    print(forecast_sku("SKU105", 90).head(14).to_string(index=False))

    print("\n=== PROJECTED INVENTORY: SKU105 @ WH006, next 30 days ===")
    proj = project_inventory_forward("SKU105", "WH006", 90)
    print(proj.head(30).to_string(index=False))
    stockout_days = proj[proj.PredictedStockout]
    print(f"\nPredicted stockout days in next 90d: {len(stockout_days)}")
    if len(stockout_days):
        print("First predicted stockout date:", stockout_days.iloc[0]["Date"].date())
