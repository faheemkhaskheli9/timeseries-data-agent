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
