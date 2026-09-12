# Architecture Notes: Time-Series Analytics Chatbot

## Pipeline

```text
Upload Data -> NL Question -> Function-Calling into Pandas Ops -> Chart + LLM Explanation
```

## Components

- CSV/JSON upload
- Natural-language questions about trends and anomalies
- Automatic plotting
- Function calling into Pandas operations
- LLM-generated explanations of findings
- Simple forecasting

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.

## Documented size limit (data profile, issue #3)

`tsda.profile.profile_dataframe` reads at most `MAX_PROFILE_ROWS` (200,000)
rows when computing missing-value percentages and numeric summary stats, so
the profile panel renders in a bounded amount of work regardless of upload
size. `row_count` in the profile always reflects the full uploaded row
count; above the cap, stats are computed on a head-sample and
`DataProfile.sampled`/`sample_size` disclose that to the UI, which shows a
caption naming the sample size instead of silently understating the data.
