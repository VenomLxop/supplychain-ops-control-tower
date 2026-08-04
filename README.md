# Global Supply Chain Operations Control Tower

🔗 **[Live Demo](https://supplychain-ops-control-tower-muedllyvhn77kaxcyseffn.streamlit.app/)** · [Data Prep Notes](data/README_data_prep.md)

*Inspired by the operational challenges faced by global consumer electronics manufacturers.*

A simulation → forecasting → optimization → decision-support pipeline for a multi-warehouse,
multi-factory supply chain network, wrapped in an interactive Streamlit dashboard.

## What it answers
- Which warehouse is projected to stock out, and when?
- What happens if a factory shuts down for N days?
- What happens if a critical component/supplier becomes unavailable?
- What happens if demand spikes 30%+ in a region?
- Given a supply shortage, how should it be allocated across warehouses to minimize cost?

## Structure
```
data/                          # cleaned + bridging-layer datasets (see data/README_data_prep.md)
simulation_engine.py            # baseline (s,S) inventory policy + 3 disruption shock types
forecasting_engine.py           # Holt's-trend demand forecast -> projected inventory
optimization_engine.py          # MILP reallocation optimizer vs. naive pro-rata baseline
dashboard.py                     # Streamlit app tying all three together
requirements.txt
```
<img width="1600" height="568" alt="Disruption Scenarios" src="https://github.com/user-attachments/assets/da8e090c-14e9-46a6-b5af-45e9647ffbe1" />

<img width="1600" height="766" alt="Network Overview" src="https://github.com/user-attachments/assets/bfab2500-0f88-48b0-b422-14174e4995b2" />

## Running it
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploying
Push this folder to a GitHub repo, then deploy free on [share.streamlit.io](https://share.streamlit.io)
by pointing it at `dashboard.py`.

## Key design decisions (worth knowing before an interview)
1. **Data bridging layer** — the source dataset had realistic entities but wasn't fully wired
   together (no BOM, no regional demand split, sparse inventory). See `data/README_data_prep.md`
   for exactly what was added/assumed and why, including a lead-time integer-rounding bug that
   was caught and fixed during the inventory panel build (93% → 1.5% stockout rate).
2. **Factory shutdown has limited impact** in this network — 6 multi-sourced factories mean no
   single factory is a real point of failure. A genuine resilience finding, kept in the dashboard
   rather than hidden, with a supplier/component disruption scenario shown alongside it as the
   scenario with real teeth (single-sourced components with no substitute).
3. **The dataset's own `Forecast` column was not used** — it achieves ~5% MAPE against demand
   that has near-zero autocorrelation and ~41% coefficient of variation, which is well beyond
   what any legitimate out-of-sample model can achieve on this kind of noise. That's consistent
   with it having been generated from same-day actuals rather than a genuine forward prediction.
   A Holt's linear-trend model (no weekly seasonality — checked, doesn't exist in the data) is
   used instead, with an honest ~40-45% MAPE.
4. **The reallocation optimizer is a MILP, not a pure LP** — an earlier linear-cost version was
   mathematically degenerate (total $ value lost is conserved regardless of allocation, when
   value-per-unit is uniform and total supply is fixed). Adding a fixed per-warehouse
   stockout-event penalty (documented assumption) gives the optimizer a genuine reason to
   concentrate an unavoidable shortage on fewer warehouses rather than spread it thin — this
   produces measurable, explainable savings (~2-5% depending on scenario parameters).
