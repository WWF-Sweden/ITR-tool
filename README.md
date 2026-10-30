# WWF ITR Tool

Calculate company and portfolio temperature scores using the
[CDP-WWF Temperature Scoring Methodology v1.5](https://wwfint.awsassets.panda.org/downloads/cdp-wwf-temperature-scoring-methodology---september-2024.pdf),
and measure portfolio coverage by SBTi-validated near-term targets.

**[Start with the shared documentation](https://wwf-sweden.github.io/ITR-tool/index.html)**
for the app, Docker, notebooks and Python.

| How you want to work | Start here |
| --- | --- |
| Try a browser demo with sample data | [Online app](https://wwf-itr.streamlit.app/) and [app setup guide](https://wwf-sweden.github.io/ITR-tool/getting_started_app.html) |
| Use a graphical interface locally | [Docker or local Streamlit](https://wwf-sweden.github.io/ITR-tool/getting_started_app.html#run-locally-with-docker) |
| Learn and adapt an analysis step by step | [Notebooks: Colab or local Jupyter](https://wwf-sweden.github.io/ITR-tool/notebooks.html) |
| Integrate scoring into your code | [Python quickstart](https://wwf-sweden.github.io/ITR-tool/getting_started.html) |

Use sample or public data in the online demo and Colab. For confidential
portfolio data, use a local environment.

## Python package

This repository contains the `wwf-itr` scoring engine, imported as `ITR`,
the notebooks and the shared Sphinx documentation. The Streamlit app is in
[WWF-ITR-APP](https://github.com/WWF-Sweden/WWF-ITR-APP).
Installing the engine alone does not install the app.

Use Python 3.10 or later in a virtual environment:

```sh
python -m pip install wwf-itr
```

For source development, clone this repository and run `python -m pip install -e .`
from its root, or use `poetry install` for the development environment.
See [Contributing](https://wwf-sweden.github.io/ITR-tool/contributing.html)
for checks, release steps and building docs.

## Data and examples

Use [Data Requirements](https://wwf-sweden.github.io/ITR-tool/DataRequirements.html)
and [Data Legends](https://wwf-sweden.github.io/ITR-tool/Legends.html) for templates,
columns, weighting inputs and SBTi CTA data. Target-based scoring and
aggregation of ready-made scores use different inputs.

The [standalone portfolio coverage notebook](examples/Simple_portfolio_coverage.ipynb)
calculates WATS coverage from a portfolio and CTA data without installing
`wwf-itr`. It does not calculate temperature scores or reconstruct historical
SBTi status.

## Repository structure

```text
.github/workflows/  GitHub Actions
docs/              Sphinx documentation sources and data templates
ITR/               Python scoring engine
examples/          Jupyter notebooks and sample data
test/              Automated tests
```

Report engine/notebook problems in [ITR-tool issues](https://github.com/WWF-Sweden/ITR-tool/issues)
and app/Docker problems in [app issues](https://github.com/WWF-Sweden/WWF-ITR-APP/issues).
Questions: ekonomi-finans@wwf.se.

See [LICENSE](LICENSE). Results are informational and are not financial or investment advice.
