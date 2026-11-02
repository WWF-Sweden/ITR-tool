Getting Started with Notebooks
==============================

Notebooks combine explanatory text, runnable Python cells and results.
Use them to follow the methodology and adapt an analysis step by step.
For a graphical interface, use :doc:`getting_started_app` instead.

Google Colab
------------

Open a Colab link below and run the cells from top to bottom, including
installation and resource-download cells. Start with **Analysis example**
for a guided explanation or **Quick temperature calculation** for a shorter
workflow. Restart the runtime if an upgraded package has already been
imported. See Google's
`notebook introduction <https://colab.research.google.com/notebooks/basic_features_overview.ipynb>`_
if you are new to Colab.

Colab runs on Google's infrastructure. Use sample or public data there;
use local Jupyter for confidential portfolios. Save or download your
results before ending the Colab session.

Local Jupyter
-------------

Install Python 3.10 or later and Git. Clone the repository:

.. code-block:: console

   git clone https://github.com/WWF-Sweden/ITR-tool.git
   cd ITR-tool

Create a virtual environment and activate it. On macOS/Linux:

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1

For the temperature-scoring notebooks, install the package and notebook tools:

.. code-block:: console

   python -m pip install wwf-itr jupyter matplotlib
   cd examples
   python -m notebook

Open a notebook in the browser and select the kernel for this environment.
Run cells in order. Starting Jupyter in ``examples`` ensures the relative
``data/`` paths refer to the supplied example files.

If you use Conda instead, from the repository root:

.. code-block:: console

   cd examples
   conda env create -f environment.yml
   conda activate ITR_getting_started
   python -m pip install matplotlib
   python -m notebook

The Conda environment installs the package from GitHub source; the pip
instructions above install a published PyPI release. Record the version
used for your analysis. The standalone portfolio coverage notebook has
its own installation cell and does not require ``wwf-itr``.

Notebook guide
--------------

.. list-table:: Examples
   :header-rows: 1
   :widths: 24 48 28

   * - Notebook
     - Purpose
     - Open
   * - 1. Analysis example
     - Guided methodology, company scores, portfolio analysis and export.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/1_analysis_example.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/1_analysis_example.ipynb>`__
   * - 2. Quick temperature calculation
     - A short target-based temperature calculation.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/2_quick_temp_score_calculation.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/2_quick_temp_score_calculation.ipynb>`__
   * - 3. What-if analysis
     - Compare scenarios and engagement assumptions.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/3_what-if-analysis.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/3_what-if-analysis.ipynb>`__
   * - 4. Portfolio aggregations
     - Explore the portfolio weighting methods.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/4_portfolio_aggregations.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/4_portfolio_aggregations.ipynb>`__
   * - 4b. Ready-made temperature scores
     - Aggregate company scores supplied by a provider.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/4b_portfolio_agg_readymade_TS.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/4b_portfolio_agg_readymade_TS.ipynb>`__
   * - 5. Reporting
     - Inspect, anonymize and export analysis data.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/5_reporting.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/5_reporting.ipynb>`__
   * - Standalone portfolio coverage
     - Calculate investment-weighted SBTi coverage from a portfolio and CTA file, without installing ``wwf-itr``. No temperature scores or historical status reconstruction.
     - `GitHub <https://github.com/WWF-Sweden/ITR-tool/blob/main/examples/Simple_portfolio_coverage.ipynb>`__ | `Colab <https://colab.research.google.com/github/WWF-Sweden/ITR-tool/blob/main/examples/Simple_portfolio_coverage.ipynb>`__

Use :doc:`DataRequirements` and :doc:`Legends` when replacing sample files.
For errors involving paths, kernels, identifiers or CTA files, see
:doc:`troubleshooting`.

The target-based examples use the fictional 360-company dataset:
``examples/data/data_provider_example_360.xlsx`` and
``examples/data/example_portfolio_360.csv``. The standalone coverage notebook
uses the portfolio from the same pair. Notebook 4b uses its separate
ready-made-score workbook, ``example_data_provider_TS.xlsx``.
Download the :download:`360-company portfolio <../examples/data/example_portfolio_360.csv>`
and :download:`provider workbook <../examples/data/data_provider_example_360.xlsx>`
if you want to inspect the input files directly.
Existing old sample files are not overwritten. Download URLs target
``main``; the new data files must be published there before Colab can
download them. To test a published review branch, change ``main`` in the
download URL to that branch.
