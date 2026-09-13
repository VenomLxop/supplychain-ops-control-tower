"""
Regenerates data/warehouse_demand_share.csv and adds a TargetZ column to
data/warehouses.csv.

Why this exists (see the dated note in README_data_prep.md for the full
writeup): the original one-shot generator (documented in section 3 below)
produced a single FIXED (SKU, Warehouse) share with no day-to-day variation,
and every warehouse used the same hardcoded Z=1.65 target service level in
simulation_engine.build_params(). Together those two choices mean every
warehouse's demand is just a scaled copy of the same national series, and
every warehouse converges to nearly the same stockout rate by construction -
not a realistic outcome. This script keeps the original Dirichlet base-share
step (it's a legitimate, documented assumption) and adds two things on top:

1. Independent day-to-day multiplicative noise per warehouse (a regional
   demand-volatility factor, std dev 5-20% and warehouse-specific), applied
   to the base share and renormalized so each SKU's shares across all 8
   warehouses still sum to 1.0 every day - total demand still ties back
   exactly to the national sales_orders.csv total.
2. A distinct TargetZ per warehouse (1.2-2.0), written as an explicit column
   in warehouses.csv - a documented assumption representing different
   regional safety-stock policies, replacing the single global Z=1.65.

A first pass with noise/Z uniformly sampled across those bands for every
warehouse only produced a 1.28%-1.95% stockout spread network-wide - real,
but not enough to show any warehouse as genuinely "at risk" on the
dashboard. A second pass tried pushing one warehouse's TargetZ down toward
(and past) 0 to compensate - even Z=-2.0 only reached 4.6%. Root cause,
diagnosed before changing anything further: S = mu*(R+LT) + Z*sigma*sqrt(R+LT).
The base term mu*(R+LT) covers ~20 days of average demand and dominates the
formula (demand CV is ~40%, but that CV only feeds the comparatively small
Z*sigma*sqrt(R+LT) safety-margin term) - Z was never going to move the
dominant term, only the margin on top of it, no matter how negative.

So RISKY_WAREHOUSES below instead gets a direct BufferMultiplier applied to
the WHOLE computed S (see BufferMultiplier column in warehouses.csv and
simulation_engine.build_params()), while keeping its TargetZ at a realistic,
non-extreme value. This is a different kind of assumption than TargetZ - not
"less safety margin" but "runs its whole stocking policy leaner" - and is
documented as such (see the dated note in README_data_prep.md for the full
writeup and the real-world justification: lowest total order volume of the
8 warehouses -> smallest market -> the one place a network would plausibly
run a leaner stocking policy).

Run from the repo root: python3 data/build_demand_share.py
Seeded with np.random.seed / np.random.default_rng(SEED) for reproducibility,
matching the "seeded for reproducibility" convention already used for
bom.csv (see README_data_prep.md section 1).
"""
import numpy as np
import pandas as pd

SEED = 42
DATA = "data"

# Deliberately leaner-stocked warehouse(s): lowest total demand volume of
# the 8 (see README_data_prep.md) - a smaller regional market a network would
# plausibly run leaner. Kept to a max of 2 by design so the healthy majority
# of the network still reads as healthy.
RISKY_WAREHOUSES = ["WH004"]
RISKY_Z = 1.0                    # realistic, not extreme - see module docstring
RISKY_BUFFER_MULTIPLIER = 0.25   # the actual risk lever - applied to the whole S, not just its margin
RISKY_NOISE_RANGE = (0.25, 0.30)


def main():
    rng = np.random.default_rng(SEED)

    products = pd.read_csv(f"{DATA}/products.csv")
    warehouses = pd.read_csv(f"{DATA}/warehouses.csv")

    skus = products["SKUID"].tolist()
    whs = warehouses["WarehouseID"].tolist()
    healthy_whs = [wh for wh in whs if wh not in RISKY_WAREHOUSES]
    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    # --- 1. Base Dirichlet share per (SKU, Warehouse) - same approach as the
    # original generator: moderate variance, not perfectly even. ---
    alpha = np.full(len(whs), 4.0)
    base_share = pd.DataFrame(
        rng.dirichlet(alpha, size=len(skus)), index=skus, columns=whs
    )

    # --- 2. Per-warehouse day-to-day noise std - documented assumption
    # representing different regional demand volatility. Healthy warehouses:
    # 5-20%. RISKY_WAREHOUSES: 25-30% (more volatile local demand on top of
    # less safety stock - a coherent combination, not just extra randomness). ---
    wh_noise_std = pd.Series(rng.uniform(0.05, 0.20, size=len(healthy_whs)), index=healthy_whs)
    for wh in RISKY_WAREHOUSES:
        wh_noise_std[wh] = rng.uniform(*RISKY_NOISE_RANGE)
    wh_noise_std = wh_noise_std.reindex(whs)

    # --- 3. Per-warehouse target service level Z - documented assumption
    # representing different regional safety-stock policies. Healthy
    # warehouses: evenly spaced across 1.2-2.0 (order shuffled with the same
    # seeded RNG) so the assumption actually produces a spread rather than
    # random draws that could cluster together. RISKY_WAREHOUSES keep a
    # realistic, non-extreme Z (RISKY_Z) - the leaner policy is expressed via
    # BufferMultiplier below instead, not by distorting Z into an
    # unrealistic/negative "safety margin". ---
    z_values = rng.permutation(np.linspace(1.2, 2.0, len(healthy_whs)))
    wh_target_z = pd.Series(np.round(z_values, 2), index=healthy_whs)
    for wh in RISKY_WAREHOUSES:
        wh_target_z[wh] = RISKY_Z
    wh_target_z = wh_target_z.reindex(whs)

    # --- 4. Per-warehouse BufferMultiplier - documented assumption applied to
    # the WHOLE computed order-up-to level S (see simulation_engine.build_params()),
    # not just its safety-margin term. 1.0 for every warehouse except
    # RISKY_WAREHOUSES, which run a deliberately leaner stocking policy. ---
    wh_buffer_mult = pd.Series(1.0, index=whs)
    for wh in RISKY_WAREHOUSES:
        wh_buffer_mult[wh] = RISKY_BUFFER_MULTIPLIER

    # One noise multiplier per (Date, Warehouse), shared across SKUs for that
    # warehouse that day (a regional demand swing, not per-SKU noise).
    noise = pd.DataFrame(
        rng.normal(loc=1.0, scale=wh_noise_std.values, size=(len(all_dates), len(whs))),
        index=all_dates, columns=whs,
    ).clip(lower=0.2)  # floor so one bad draw can't zero out a warehouse's share

    frames = []
    for sku in skus:
        raw = base_share.loc[sku].values * noise.values          # (n_days, n_wh)
        norm = raw / raw.sum(axis=1, keepdims=True)                # renormalize -> sums to 1/day
        frames.append(pd.DataFrame(norm, index=all_dates, columns=whs)
                      .rename_axis("Date").reset_index()
                      .melt(id_vars="Date", var_name="WarehouseID", value_name="DemandShare")
                      .assign(SKUID=sku))

    demand_share = pd.concat(frames, ignore_index=True)[["Date", "SKUID", "WarehouseID", "DemandShare"]]
    demand_share.sort_values(["SKUID", "WarehouseID", "Date"], inplace=True)
    demand_share.to_csv(f"{DATA}/warehouse_demand_share.csv", index=False)

    warehouses["TargetZ"] = warehouses["WarehouseID"].map(wh_target_z).values
    warehouses["BufferMultiplier"] = warehouses["WarehouseID"].map(wh_buffer_mult).values
    warehouses.to_csv(f"{DATA}/warehouses.csv", index=False)

    print(f"Wrote {len(demand_share)} rows to warehouse_demand_share.csv")
    print("\nPer-warehouse noise std (5-20% band):")
    print(wh_noise_std.round(4))
    print("\nPer-warehouse TargetZ (written to warehouses.csv):")
    print(wh_target_z)
    print("\nPer-warehouse BufferMultiplier (written to warehouses.csv):")
    print(wh_buffer_mult)
    # sanity check: shares still sum to 1.0 per (SKU, Date)
    check = demand_share.groupby(["SKUID", "Date"])["DemandShare"].sum()
    print(f"\nShare-sum sanity check: min={check.min():.10f} max={check.max():.10f}")


if __name__ == "__main__":
    main()
