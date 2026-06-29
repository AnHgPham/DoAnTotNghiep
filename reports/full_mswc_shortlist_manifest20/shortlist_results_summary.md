# Full MSWC Shortlist Manifest20 Results

Source date checked: 2026-06-01 15:36 ICT.

## Run Status

| Pipeline | Train | Dev30 | Test100 | Started | Finished |
|---|---|---|---|---|---|
| DSCNN-L + PCEN + GE2E | ok | ok | ok | 2026-05-31 10:34:11 | 2026-05-31 14:54:04 |
| EdgeSpotFull T4 + PCEN + GE2E | ok | ok | ok | 2026-05-31 14:54:04 | 2026-05-31 22:14:45 |

## Evaluation Metrics

All values are mean +/- std. The main operating point is ACC@1%FAR.

| Pipeline | Split | Runs | ACC@1%FAR | FRR@1%FAR | ACC@5%FAR | FRR@5%FAR | AUC | EER | KW-ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | dev30 | 30 | 82.69 +/- 1.26 | 52.26 +/- 4.21 | 85.98 +/- 0.79 | 30.67 +/- 2.43 | 91.96 +/- 0.68 | 15.89 +/- 1.03 | 87.96 +/- 1.13 | 76.40 +/- 1.39 |
| DSCNN-L + PCEN + GE2E | test100 | 100 | 82.10 +/- 0.87 | 55.48 +/- 2.94 | 86.05 +/- 0.66 | 31.38 +/- 2.28 | 91.57 +/- 0.58 | 16.25 +/- 0.86 | 88.90 +/- 1.25 | 75.90 +/- 1.16 |
| EdgeSpotFull T4 + PCEN + GE2E | dev30 | 30 | 79.35 +/- 1.00 | 62.75 +/- 3.49 | 82.60 +/- 0.63 | 40.07 +/- 2.19 | 88.55 +/- 0.73 | 18.90 +/- 1.03 | 82.75 +/- 1.21 | 72.40 +/- 1.34 |
| EdgeSpotFull T4 + PCEN + GE2E | test100 | 100 | 79.58 +/- 0.91 | 63.45 +/- 3.18 | 83.06 +/- 0.82 | 40.01 +/- 2.96 | 87.22 +/- 0.75 | 20.40 +/- 1.01 | 83.01 +/- 1.49 | 70.46 +/- 1.30 |

## Interpretation

- DSCNN-L + PCEN + GE2E is the strongest accuracy-oriented shortlist model.
- EdgeSpotFull T4 + PCEN + GE2E is smaller and remains the edge-oriented candidate, but it trails DSCNN-L by 2.52 percentage points on GSC-test100 ACC@1%FAR.
- The gap is larger on F1: DSCNN-L reaches 75.90%, while EdgeSpotFull T4 reaches 70.46%.
- EdgeSpotFull T4 still has thesis value if the argument emphasizes compactness and edge deployment; DSCNN-L is the better candidate if the primary objective is highest test100 accuracy.

## Evidence Files

- Raw summary: `reports/full_mswc_shortlist_manifest20/raw/full_mswc_shortlist_manifest20_e20_ep200_runs.tsv`
- DSCNN dev30 JSON: `reports/full_mswc_shortlist_manifest20/raw/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_k10_results.json`
- DSCNN test100 JSON: `reports/full_mswc_shortlist_manifest20/raw/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/test100/gsc_edgespot_exact_k10_results.json`
- EdgeSpot dev30 JSON: `reports/full_mswc_shortlist_manifest20/raw/full_mswc_shortlist_manifest20_e20_ep200/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_k10_results.json`
- EdgeSpot test100 JSON: `reports/full_mswc_shortlist_manifest20/raw/full_mswc_shortlist_manifest20_e20_ep200/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/test100/gsc_edgespot_exact_k10_results.json`

## Recommended Next Step

Use these two test100 rows as the current shortlist evidence. Do not launch another broad matrix immediately. The next experiment should be one targeted attempt to improve EdgeSpotFull T4, because it is the smaller model and currently trails DSCNN-L. Candidate next run:

- EdgeSpotFull T4 + PCEN + GE2E
- Longer or richer training than manifest20 phase, preferably after building a larger capped manifest such as max50.
- Keep the same GSC-test100 protocol so the result is comparable.
