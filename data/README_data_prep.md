# Data Prep Notes — Bridging Layer

Original file: `Global_Operations_Control_Tower_Dataset_v3.xlsx`
This folder = original tables (unchanged, converted to CSV) + 4 new tables that close the
gaps needed for the simulation engine.

## What was added and why

### 1. `bom.csv` — Bill of Materials (SKU → Component)
The original data had 40 components (via PurchaseOrders) and 30 SKUs but no link between
them, so a supplier disruption had no way to cascade to specific products.
- Each SKU assigned 4-6 components (mean 4.9), `QtyPerUnit` 1-3, seeded for reproducibility.
- **Use this to answer**: "Supplier X goes down → which components are short → which SKUs
  are affected → which factories/warehouses feel it."

### 2. `shipments_enriched.csv` — Shipments + Date/SKUID/Qty/ArrivalDate
Original Shipments had Origin/Destination/Mode/LeadTimeDays/Status but no SKU, quantity, or
date — so a delayed shipment couldn't be traced to a specific product shortage.
- SKU assigned per shipment weighted by that origin factory's actual historical production
  mix (from ProductionOrders).
- Qty sampled as 30-70% of that SKU's average daily production at that factory (a plausible
  partial shipment size).
- ShipDate spread across the year per route; `ArrivalDate = ShipDate + LeadTimeDays`.
- **Note**: this table is best used for the logistics/map layer (routes, delays, mode mix) —
  it is *not* the source of truth for inventory (see below), since with only ~8,000
  shipments across 48 routes × 30 SKUs, replenishment per SKU is too infrequent to model
  realistic day-to-day stock levels directly.

### 3. `warehouse_demand_share.csv` — regional demand split
SalesOrders in the original data is a single global daily quantity per SKU with no
warehouse/region attached. To simulate warehouse-level stockouts you need demand split
by location.
- Generated via Dirichlet distribution per SKU across the 8 warehouses (moderate variance,
  not perfectly even — some warehouses are naturally higher/lower demand for a given SKU).
- **This is a modeling assumption, not observed data** — flag this explicitly if asked in
  an interview: "I allocated national demand to warehouses using a documented synthetic
  share, since the source data didn't include regional demand splits."

### 4. `inventory_daily.csv` — full daily inventory panel (365 days × 8 warehouses × 30 SKUs = 87,600 rows)
Original Inventory sheet had only 2,920 sparse snapshot rows — nowhere near enough for a
"which warehouse stocks out next week" model.
- Built via a **periodic-review, order-up-to-level policy** (R=7 day review cycle,
  95% target service level, lead time = average route lead time to that warehouse),
  simulated forward day-by-day — this is standard inventory theory (the same logic behind
  real-world reorder-point systems), not just interpolation.
- Demand per warehouse-SKU pulled from actual SalesOrders × `warehouse_demand_share`.
- **Baseline result: 1.5% stockout rate**, spread across combos (range 0.5%-2.5%) —
  realistic for a healthy-but-not-perfect network. This gives headroom to show a
  disruption scenario meaningfully *increasing* stockout risk, rather than starting from
  an already-broken or already-perfect baseline.
- **Known bug caught during build**: lead times were non-integer (e.g. 13.47 days), which
  silently broke the delivery-arrival matching (float ≠ int day index) and caused 93%+
  stockouts. Fixed by rounding lead time to whole days. Worth mentioning in interviews as
  an example of catching a data-integrity bug before it corrupted downstream analysis.

### 5. `warehouse_demand_share.csv` v2 + `warehouses.csv` TargetZ/BufferMultiplier — regional demand variance (2026-09-13)
Follow-up to section 3 above. The original demand-share table gave each warehouse a
FIXED proportional share of one national demand series, with no independent day-to-day
noise, and `simulation_engine.build_params()` used the same hardcoded `Z=1.65` target
service level for every warehouse. Together those two choices meant every warehouse's
demand was just a scaled copy of the same national series and every warehouse converged
to nearly the same stockout rate (~1.4%) by construction — not realistic, and it made the
dashboard's network map show all-green regardless of which warehouse you looked at.

`data/build_demand_share.py` (new — this repo previously had no generator script for any
derived data file; see this new script for the full generation logic) regenerates
`warehouse_demand_share.csv` in place, keeping the original Dirichlet base-share step
(still a legitimate, documented assumption) and adding:
- **Independent day-to-day multiplicative noise per warehouse** (mean 1.0, warehouse-specific
  std dev ~10-17% for the 7 "healthy" warehouses), renormalized so each SKU's shares across
  all 8 warehouses still sum to 1.0 every day — total demand still ties back exactly to the
  national `sales_orders.csv` total.
- **A distinct `TargetZ` per warehouse** (1.2-2.0 for the 7 healthy warehouses), written as an
  explicit column in `warehouses.csv` — a documented assumption representing different
  regional safety-stock policies, replacing the single global `Z=1.65`.

**A lever I tried and discarded**: my first pass used TargetZ alone (uniformly sampled
1.2-2.0, later pushed as low as -2.0 for one warehouse) to try to create a warehouse that
reads as genuinely "at risk" (>5% stockout) on the dashboard map. It didn't work — even
Z=-2.0 only reached 4.6%. Root cause: `S = mu*(R+LT) + Z*sigma*sqrt(R+LT)`. The base term
`mu*(R+LT)` covers ~20 days of average demand and dominates the formula; Z only scales the
comparatively small safety-margin term on top of it, so no amount of Z tuning was ever
going to create real risk — I was tuning the wrong knob. (The healthy-warehouse TargetZ
spread of 1.2-2.0 is still used as-is — it produces a real, if modest, 1.3%-1.9% stockout
spread across those 7 warehouses, which is honest and worth keeping.)

**The actual fix**: a new `BufferMultiplier` column in `warehouses.csv` (default 1.0),
applied in `build_params()` to the WHOLE computed S, not just its safety-margin term. WH004
(the lowest total order volume of the 8 warehouses — the smallest market, and the one place
a network would plausibly run a leaner stocking policy) gets `TargetZ=1.0` (a realistic,
non-extreme safety margin) plus `BufferMultiplier=0.25` and elevated demand noise (~29%,
vs ~10-17% for the others) — a warehouse that both runs lean AND faces more volatile local
demand, which is a coherent combination, not just stacked randomness.
- **Result**: WH004 lands at ~7% stockout rate (up from ~1.4% before this change), while the
  other 7 warehouses stay in a healthy 1.3%-1.9% band. On the dashboard's network map, WH004
  now shows red (>5% threshold) for 29 of 30 SKUs, instead of every warehouse showing green.
- **This is a documented assumption, not observed data** — `TargetZ`, `BufferMultiplier`, and
  the per-warehouse noise std are all explicit, seeded (`np.random.default_rng(42)`), and
  overridable, not hidden constants. Forecast MAPE is unaffected (43.7%/53.2%/4.9%, unchanged)
  since this only touches how national demand is *split and buffered* across warehouses, not
  the national demand signal itself.

## What to tell an interviewer about this step
"The raw synthetic dataset had realistic entities but wasn't fully wired together —
no BOM, no regional demand split, no continuous inventory. I built a bridging layer using
a standard inventory policy simulation rather than just interpolating gaps, which is both
more defensible and gives me a natural 'baseline vs. disruption' comparison for the
scenario engine."
