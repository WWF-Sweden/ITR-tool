Using the App
=============

Start the online or local app using :doc:`getting_started_app`. The
workflow below applies to both. Use the online demo only with sample or
public data; use a local installation for confidential data.

First analysis with sample data
-------------------------------

1. In the sidebar, select **Sample Data** and click **Load Data**.
2. Keep the initial settings: **MID**, **Scope 1+2**, **Scope 1+2+3**,
   **WATS (Weighted Average)** and **SBTi Factor** of ``1.0``.
3. Review the portfolio, fundamental data and target data in the main view.
4. Click **Run Analysis**. Open **Portfolio Overview** to inspect scores
   by timeframe and scope. Check which combination the overview chart shows.
5. Explore **Hotspot Analysis** and **Company Analysis**, then use
   **Export Data** to download the results.

Portfolio coverage is optional and is not selected initially. To include
it, enable **Calculate portfolio coverage** and choose CTA data as described
below. A temperature score and SBTi portfolio coverage answer different
questions: an ambitious target does not by itself establish SBTi validation.

Upload and edit your own data
-----------------------------

Select **Upload Custom Data**. Upload both files before clicking **Load Data**:

* A provider Excel workbook with sheets named ``fundamental_data`` and
  ``target_data``. Start with the
  :download:`provider template <DataProviderTemplate.xlsx>`.
* A portfolio CSV or Excel file with ``company_name``, ``company_id`` and
  ``investment_value``. Start with the
  :download:`portfolio template <PortfolioTemplate.csv>`.

Use the same ``company_id`` in the portfolio and both provider sheets.
Use a common currency for investment values and financial denominators.
See :doc:`DataRequirements` and :doc:`Legends` for conditional requirements,
identifiers, units and accepted values. A CSV must use comma-separated
columns; save uploaded CSV files as UTF-8.

Use **Edit Portfolio**, **Edit Fundamentals** or **Edit Targets**
in the review tabs and click **Apply changes** to accept edits. Review
any data-cleaning or validation messages before running the analysis.
Changes to scoring inputs or parameters require **Run Analysis** again.
To load a different source, use **Reload / Change Source**.

Use pre-scored data (4b format)
-------------------------------

Choose **Pre-scored Data (4b format)** to aggregate temperature scores
already calculated by a data provider. Upload one Excel workbook, using
its first sheet, with one row per company, scope and timeframe combination.
Start with the :download:`ready-made scores template <ReadyMadeScoresTemplate.xlsx>`.

The required columns are ``company_id``, ``company_name``,
``investment_value``, ``scope``, ``time_frame`` and ``temperature_score``.
For example, use ``S1S2`` for a combined scope and ``MID`` for a timeframe.
Select scopes and timeframes present in the file. WATS uses investment
values; other methods also require the relevant emissions and financial
columns listed in :doc:`DataRequirements`.

The app aggregates these scores without recalculating company targets.
Portfolio coverage, SBTi Factor adjustments, target editing and what-if
scenarios are not available in this mode. Additional columns are limited
to those supported by the pre-scored importer; use Python for custom
extensions. For a worked example, see notebook 4b in :doc:`notebooks`.

Analysis parameters and portfolio coverage
------------------------------------------

Select one or more **Timeframes** and **Scopes**, and an **Aggregation
Method**. The available scopes are S1, S2, S1+S2, S3 and S1+S2+S3.
The available methods are WATS, TETS, MOTS, EOTS, ECOTS, AOTS and ROTS.
See :doc:`FunctionalOverview` for their interpretation.

**SBTi Factor** defaults to ``1.0``: target-based scores are used
regardless of validation status. Values below ``1.0`` blend scores for
non-validated companies towards the default temperature score. This
setting affects temperature scores; it is not the portfolio coverage
percentage.

CTA controls appear when **Calculate portfolio coverage** is enabled or
SBTi Factor differs from ``1.0``:

* **Auto-download from SBTi** uses a cached file for up to one week before
  attempting a refresh. **Update CTA from SBTi** forces a refresh.
* **Upload custom file** uses a CTA Excel file for the active session.
* **Offline mode** uses cached or bundled data without downloading.
  Uploaded CTA files also work offline. Bundled fallback data may be old.

Check the displayed CTA file, modification time and any fallback warnings.
A modification time is not a historical status date. Invalid or unavailable
CTA data is reported as an error, rather than silently treated as zero
coverage. CTA status labels and identifier matching are explained under
:ref:`sbti-status`.

Results, scenarios and export
-----------------------------

**Portfolio Overview** shows the aggregated scores by timeframe and scope.
**Hotspot Analysis** provides filters and chart options, and **Company
Analysis** shows individual scores and contributing companies. Read chart
labels carefully: filtered or descriptive charts do not necessarily use
the same weighting as the selected portfolio aggregation method.

In **What-If Scenarios**, choose a scenario type, engagement type and
comparison timeframe/scope. Select engagement candidates when applicable
and click **Run Scenario Analysis**. These results illustrate changed
target assumptions; they do not update the observed SBTi status.

In **Export Data**, choose:

* **Download Portfolio CSV** for the portfolio with calculated scores.
* **Download Scores CSV** for portfolio-level aggregated scores.
* **Generate Excel File**, then **Download Excel File**, for a workbook
  with ``Portfolio Scores``, ``Aggregated Scores`` and ``Original Portfolio``.

The Excel export is not a complete archive of provider inputs, CTA data
and every scenario. Keep the original inputs, selected CTA snapshot,
parameters and software versions with the results when reproducibility
matters.

Save and reload datasets
------------------------

For target-based input data, expand **Save data to local database**,
enter a dataset name and optionally a description, then click **Save**.
This saves the current portfolio, fundamentals and target tables. It
does not archive all calculation outputs or the selected CTA file.

Choose **Load from Database**, select the saved dataset and click
**Load Data** to reuse it. Saving under an existing name updates that
dataset. **Delete this dataset** removes it from the database.

For Docker, the database persists through the mounted ``data`` folder
described in :ref:`app-docker`. For direct Python execution, it is in
the app repository's ``data`` folder. In the public cloud demo, storage
is on the cloud host and may be lost across restarts; it is not a
private database on your computer.
