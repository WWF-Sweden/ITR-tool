Data Requirements
=================

Choose the input workflow
-------------------------

.. list-table:: Files for each analysis
   :header-rows: 1
   :widths: 25 47 28

   * - Analysis
     - Required inputs
     - Start with
   * - Calculate temperature scores from targets
     - Portfolio plus company fundamentals/emissions and target data.
     - :download:`Portfolio CSV <PortfolioTemplate.csv>` and :download:`provider workbook <DataProviderTemplate.xlsx>`.
   * - Aggregate existing company temperature scores
     - One table containing scores, holdings, scopes and timeframes, plus data needed for the selected weighting method.
     - :download:`Ready-made scores workbook <ReadyMadeScoresTemplate.xlsx>` and notebook 4b in :doc:`notebooks`.
   * - Calculate standalone portfolio coverage (WATS)
     - Portfolio with investment values and LEI/ISIN identifiers, plus SBTi CTA data. No emissions or target workbook.
     - The standalone coverage notebook in :doc:`notebooks`.

CTA data is also needed for the app's optional portfolio coverage or
SBTi Factor adjustment. The app's target-based screen expects both
portfolio and provider inputs even if coverage is your main interest;
use the standalone notebook or Python coverage example for portfolio-only
analysis.

Portfolio data
--------------

Supply ``company_name``, ``company_id`` and ``investment_value``. Use
the same company ID in the portfolio, fundamentals and target sheets;
the identifier need not be an ISIN. Keep IDs as text to preserve leading
zeros. Use the same currency across holdings and financial denominators.
For a straightforward WATS analysis use non-negative values and a
positive total investment value.

``company_isin`` and ``company_lei`` are optional model fields, not
universal requirements for temperature scoring. Supply reliable LEI or
ISIN identifiers when using CTA matching. Holdings without a usable
identifier remain unmatched. See :ref:`sbti-status` for the lookup order.

The standalone coverage notebook requires one row per ``company_id``;
combine multiple holdings in the same company before using it.
For custom grouping in Python, put additional categories in ``user_fields``
as a dictionary; see :doc:`Legends`.

Company data
------------

For target-based analysis, keep the provider template's sheet names and
columns, including columns whose cells may be blank:

* ``fundamental_data``: company name, ID, ISIC classification, emissions,
  financial values and optional grouping categories.
* ``target_data``: company ID, target type, scopes, relevant coverage
  and ambition, base/end years and supporting target fields.

Model-level optionality does not mean every column can be removed from
an Excel workbook: the Excel loader reads several columns directly.
Use the template and :doc:`Legends` rather than creating a minimal
workbook by deleting apparently unused columns.

Fundamental data
~~~~~~~~~~~~~~~~

The ISIC classification maps targets to regression models. Emissions
are used to combine scope scores and for emissions-weighted aggregation.
For uploaded Excel provider files, supply separate ``ghg_s1`` and
``ghg_s2`` when known: the loader computes their sum for ``ghg_s1s2``.
Do not rely on a combined-only value being retained by this loader.
Use tonnes of CO2 equivalent and consistent reporting periods.

.. list-table:: Aggregation inputs
   :header-rows: 1
   :widths: 15 40 45

   * - Method
     - Weighting
     - Additional company data
   * - WATS
     - Portfolio investment values
     - No company financial denominator for weighting. Emissions may still be needed for combined scope scores.
   * - TETS
     - Total emissions
     - Emissions for the requested scopes; no company financial denominator.
   * - MOTS
     - Owned emissions using market capitalization
     - Relevant emissions and ``company_market_cap``.
   * - EOTS
     - Owned emissions using enterprise value
     - Relevant emissions and ``company_enterprise_value``.
   * - ECOTS
     - Owned emissions using enterprise value plus cash
     - Relevant emissions, ``company_enterprise_value`` and ``company_cash_equivalents``.
   * - AOTS
     - Owned emissions using total assets
     - Relevant emissions and ``company_total_assets``.
   * - ROTS
     - Owned emissions using revenue
     - Relevant emissions and ``company_revenue``.

Ownership-based methods also use investment values. Required denominators
must be valid for every included company; missing data should not be
replaced with zero merely to bypass validation.

Target data
~~~~~~~~~~~

Provide public GHG reduction targets with type, scope, relevant scope
coverage, ambition and base/end years. Enter fractions as decimals:
70% is ``0.7``, not ``70``. Supply base-year emissions for relevant
scope splitting and combination. Scope 3 targets may specify categories
1–15 or a headline target; see :doc:`Legends`.

Intensity metrics must map to the accepted categories in :doc:`Legends`.
The special ``T_score`` target type is distinct from the ready-made
portfolio-score workflow below. See the
`CDP-WWF methodology <https://wwfint.awsassets.panda.org/downloads/cdp-wwf-temperature-scoring-methodology---september-2024.pdf>`_
for its meaning. Invalid targets can be skipped; review validation output
and default-score use before interpreting results.

Ready-made company scores
-------------------------

The app's 4b importer reads the first sheet of an Excel file. Required
columns are ``company_id``, ``company_name``, ``investment_value``,
``scope``, ``time_frame`` and ``temperature_score``. Use scope codes
``S1``, ``S2``, ``S1S2``, ``S3`` or ``S1S2S3`` and timeframe codes
``SHORT``, ``MID`` or ``LONG`` (case-insensitive). Each row represents
one company/scope/timeframe combination. Repeat the holding's investment
value across its different combinations, not as multiple holdings within
the same combination.

Additional emissions/financial columns depend on the weighting method
above. Supported grouping fields include sector, region, country and
industry levels. The app drops unsupported extra columns. This workflow
does not recalculate targets and does not offer coverage or scenarios in
the app. Python aggregation also needs ``temperature_results``; notebook
4b shows how to prepare it. See :doc:`FunctionalOverview` for interpretation.

.. _sbti-status:

SBTi status
-----------

The current per-company Companies Taking Action (CTA) Excel file is
downloaded from
https://sb-ti.files.svdcdn.com/staging/files/companies-excel.xlsx.
Its required columns are ``company_name``, ``isin``, ``lei`` and
``near_term_status``. This is different from the older
``Action``/``Target``/``Date Published`` schema.

``Validated Targets`` counts towards coverage; the previous label
``Targets set`` is also accepted. ``Commitment to Set Targets`` (previously
``Committed``), ``Commitment Removed`` and empty status cells do not count
as validated. Status matching ignores capitalization and surrounding
whitespace. Missing required columns or unknown nonempty statuses raise
an error.

Companies are matched using LEI when available and usable (more than three
characters), otherwise ISIN. An unmatched usable LEI does not trigger a
second attempt with ISIN. Validate identifiers carefully: no match is not
proof that a company has no approved targets. Matched validated companies
receive 100 and other holdings 0; WATS weights these by investment value.

CTA storage and reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python package caches downloads at
``~/.itr-tool/current-Companies-Taking-Action.xlsx``, normally refreshing
after one week. Failed downloads can leave it using an existing cache or
bundled fallback data that may be outdated. To select a snapshot, pass
``cta_file_path`` to ``PortfolioCoverageTVP``.

The app uses ``data/cta/current-Companies-Taking-Action.xlsx`` by default,
or ``ITR_CTA_PATH`` when configured. In Docker this persists through the
``data`` volume. The app displays the selected file and modification time,
supports uploaded CTA files and provides update/offline controls; see
:doc:`using_app`.

The standalone coverage notebook downloads directly on each requested
online run or reads a chosen local snapshot. It has no automatic cache
or fallback; a validated download is saved to ``data/cta_download.xlsx``.

Keep the CTA snapshot and software versions with your results. Current
CTA status does not reconstruct historical status, and file modification
time is not a validation or historical reporting date.
