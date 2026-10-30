# Notebook sample data

The target-based notebooks and the Python quickstart use this pair:

- `example_portfolio_360.csv`: 360 fictional portfolio companies and investment values.
- `data_provider_example_360.xlsx`: company fundamentals and 1,668 target rows for the same companies.

Company names, financial data, emissions and targets are fictional.
Public LEI/ISIN identifiers are reused to demonstrate SBTi CTA matching;
matched statuses are not claims about the fictional companies. Treat all
results as illustrations. CTA coverage depends on the selected snapshot.

Notebook 4b has a different input format and continues to use
`example_data_provider_TS.xlsx`, containing ready-made scores.
The smaller `example_portfolio.csv` and `data_provider_example.xlsx` remain
available for compatibility but are no longer the target-based notebook defaults.

The notebooks download missing default files from this folder on GitHub's
`main` branch. Publish the notebooks and new data files together. For local
use, start Jupyter from `examples` so `data/` resolves to this directory.
