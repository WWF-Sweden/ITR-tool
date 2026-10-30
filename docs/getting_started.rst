Getting Started Using Python
============================

Use the ``wwf-itr`` package to integrate temperature scoring into your own
analysis. The Python import name is ``ITR``. For a graphical interface,
use :doc:`getting_started_app`; for guided runnable examples, use
:doc:`notebooks`.

Installation
------------

Use Python 3.10 or later. Python 3.11 is a practical starting point.
Create and activate an environment, for example on macOS/Linux:

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1

Install the published package:

.. code-block:: console

   python -m pip install wwf-itr

For source development, clone
``https://github.com/WWF-Sweden/ITR-tool.git`` and install with
``python -m pip install -e .`` from the repository root. See
:doc:`contributing` for development and documentation dependencies.

.. _python-code:

First temperature score calculation
-----------------------------------

Save the following as ``first_analysis.py`` in your working folder and
run ``python first_analysis.py``. The script downloads two public sample
files only when they are absent, calculates all scopes for the mid-term
timeframe, prints portfolio scores, and writes company results to CSV.
The example uses the same fictional 360-company dataset as the notebooks;
see :doc:`notebooks` for direct downloads. Sample data is illustrative;
scores can change with the calendar year
because target timeframes are forward-looking.

.. literalinclude:: examples/first_analysis.py
   :language: python

``company_scores`` has one row per calculated company/scope/timeframe
combination, including ``temperature_score`` and underlying data.
``portfolio_scores`` contains the weighted results and contributions.
Use ``portfolio_scores.model_dump()`` for a dictionary representation.
Review skipped-target messages before relying on the outputs. See
:doc:`FunctionalOverview` for weighting and default-score interpretation.

Replace the two input paths with your own files using
:doc:`DataRequirements`. Include prerequisite scopes when calculating
combined scopes; this example explicitly requests all five.
For grouped results, pass a list such as ``grouping=["sector"]`` to
``TemperatureScore``. For more analysis and export examples, use
:doc:`notebooks`.

Portfolio coverage
------------------

SBTi portfolio coverage can be calculated from a portfolio file without
calculating temperature scores. The following example uses WATS and the
sample portfolio downloaded above:

.. code-block:: python

   import pandas as pd
   from ITR.portfolio_aggregation import PortfolioAggregationMethod
   from ITR.portfolio_coverage_tvp import PortfolioCoverageTVP

   portfolio = pd.read_csv(
       "data/example_portfolio_360.csv",
       dtype={"company_id": str, "company_isin": str, "company_lei": str},
   )
   coverage_calculator = PortfolioCoverageTVP()
   coverage = coverage_calculator.get_portfolio_coverage(
       portfolio.copy(), PortfolioAggregationMethod.WATS
   )
   print(f"Portfolio coverage: {coverage:.2f}%")

The result is already a percentage from 0 to 100. To use a specific CTA
snapshot, construct ``PortfolioCoverageTVP(cta_file_path="data/my-cta.xlsx")``.
See :ref:`sbti-status` for status matching, caches and fallback behaviour.
Keep input snapshots and record ``ITR.__version__`` with the analysis.

If you only need investment-weighted coverage and do not want to install
the scoring package, use the standalone coverage notebook in
:doc:`notebooks`. It uses pandas directly and has its own CTA loading rules.

Google Colab
------------

The notebook instructions have moved to :doc:`notebooks`. This section
is retained for existing links to ``getting_started.html#google-colab``.

Jupyter Notebooks
-----------------

For local setup and the complete notebook guide, see :doc:`notebooks`.
This section retains the existing ``#jupyter-notebooks`` link.
