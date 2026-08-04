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

## What to tell an interviewer about this step
"The raw synthetic dataset had realistic entities but wasn't fully wired together —
no BOM, no regional demand split, no continuous inventory. I built a bridging layer using
a standard inventory policy simulation rather than just interpolating gaps, which is both
more defensible and gives me a natural 'baseline vs. disruption' comparison for the
scenario engine."
