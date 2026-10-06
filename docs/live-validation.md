# Live validation — optimized NVIDIA workflow

Validated on 2026-10-06 with actual Gemini generation, SEC filings and yfinance. No demo nodes were used.

| Run | Duration | Supported citations | Outcome |
|---|---:|---:|---|
| Earlier workflow | 169.31 s | 0 / 5 | Completed, but output stopped at Company Overview and verification had false rejections |
| First optimization | 100.86 s | 28 / 33 | Full report; batched verification and local planning/analysis |
| Final fast workflow | 71.80 s | 24 / 32 | Full nine-section report; local planning, analysis and evidence-review contract |

Final run ID: `7d6a12ef-7faa-478b-9900-24cecd3a3332`. Model: `gemini-3-flash-preview`. The final measured run is about 57.6% faster than the earlier run. These are individual runs, not a controlled latency or accuracy benchmark. Different model outputs and provider load affect comparisons.

## Evidence and checks

- Downloaded the actual latest 10-Q and 10-K, current market snapshot, and SEC XBRL facts.
- Calculated ten financial ratios in Python.
- Produced all nine required sections.
- Supported 24 of 32 cited claims; 8 remain flagged.
- Made one structured provider request for 22 uncertain claim judgments; seven financial observations were supported locally.
- Rechecking the previous five saved claims supports four directly from the recorded financial fields. The earnings inference remains unverified.
- Regression checks cover wrong amounts, issuer identity, percent units, negation, forecasts, batching, budget accounting, and evidence-sensitive cache invalidation.

## Changes that reduce work

- Local plans and arithmetic summaries replace redundant model calls.
- Fast mode reviews actual risk excerpts under an explicit contract; deep mode retains additional model review.
- Semantic judgments are batched; determinate judgments can be reused across revisions only with unchanged evidence and issuer context.
- A new full report is requested only for a concrete evidence-backed correction.
- Market info and price history are reused inside a run.
- Report generation uses bounded thinking and rejects truncated or incomplete output.

## Remaining limits

- Source support is not a guarantee of truth or investment accuracy.
- Compound observations, forecasts, inferred valuations and missing evidence remain subject to abstention.
- The last small changes preserve calculation-period context and retrieval timestamps and explain uncertain semantic judgments; they do not retroactively alter the saved run counts above.

[Open the full live report](../reports/NVDA_live_research.md). Private audit snapshots remain in `audit_logs/`.
