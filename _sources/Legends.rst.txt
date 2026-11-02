Data Legends
============

Use these definitions with the downloadable templates in
:doc:`DataRequirements`. Required model fields and conditional calculation
requirements are different from the columns an Excel loader expects:
keep the provider template's full column structure, using blank cells
only where permitted. This page is the maintained field reference;
the older Excel legend is superseded by this reference.

Input Data
----------

Portfolio Data
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Format / requirement
     - Meaning
   * - ``company_name``
     - Text; required.
     - Company display name.
   * - ``company_id``
     - Text; required.
     - Identifier shared across portfolio, fundamentals and targets. Preserve leading zeros.
   * - ``investment_value``
     - Number; required.
     - Holding value, in a common currency. Use non-negative values and a positive total for the simplified WATS workflow.
   * - ``company_isin``
     - Text; optional model field.
     - ISIN for CTA matching when a usable LEI is absent. Provide an identifier when status matching is needed.
   * - ``company_lei``
     - Text; optional model field.
     - Preferred CTA matching identifier when usable; no ISIN retry after an unmatched usable LEI.
   * - ``engagement_target``
     - Boolean; optional, defaults to false.
     - Marks selected companies for the selective-engagement scenario.
   * - ``user_fields``
     - Dictionary; optional in Python.
     - Additional grouping categories, e.g. {"strategy": "equity"}. Keys are flattened into portfolio columns. This is not a guarantee that arbitrary columns survive the app's pre-scored importer.

Fundamental Data
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Format / requirement
     - Meaning
   * - ``company_name``, ``company_id``
     - Text; required model fields.
     - Company name and shared identifier.
   * - ``isic``
     - Text; required model field.
     - ISIC classification for mapping to regression sectors; preserve the template format.
   * - ``country``, ``region``, ``sector``
     - Text; optional.
     - Categories for analysis and grouping.
   * - ``industry_level_1`` through ``industry_level_4``
     - Text; optional.
     - Industry grouping categories.
   * - ``ghg_s1``, ``ghg_s2``, ``ghg_s3``
     - Numbers in tCO2e; conditional.
     - Annual emissions by scope. Needed for relevant scope combinations and emissions-weighted aggregation. Unknown is not the same as zero.
   * - ``ghg_s1s2``
     - Number in tCO2e.
     - Combined S1+S2 emissions. ExcelProvider recomputes this from separate S1 and S2 values; a combined-only input is not reliably retained.
   * - ``ghg_s3_1`` through ``ghg_s3_15``
     - Numbers in tCO2e; conditional.
     - Scope 3 category emissions used when combining category-specific targets.
   * - ``company_revenue``
     - Monetary value; required for ROTS.
     - Revenue in the same currency as investment values.
   * - ``company_market_cap``
     - Monetary value; required for MOTS.
     - Market capitalization. Also used in some notebook ownership displays.
   * - ``company_enterprise_value``
     - Monetary value; required for EOTS/ECOTS.
     - Enterprise value.
   * - ``company_cash_equivalents``
     - Monetary value; required for ECOTS.
     - Cash equivalents added to enterprise value.
   * - ``company_total_assets``
     - Monetary value; required for AOTS.
     - Total assets.

Target Data
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Format / requirement
     - Meaning
   * - ``company_id``
     - Text; required.
     - Links the target to its company.
   * - ``company_name``
     - Text; optional in the target sheet.
     - Display information. The fundamental sheet still requires the company name.
   * - ``target_type``
     - Text; required.
     - Use Absolute, Int_to_abs, Intensity, T_score or Other as appropriate. Eligibility depends on type and supporting fields; Other does not create a scored reduction target.
   * - ``intensity_metric``
     - Required for intensity targets.
     - Revenue, Product, Cement, Oil, Steel, Aluminum, Power or Other. Preserve the spelling; Power refers to power generation.
   * - ``scope``
     - Required scope code.
     - S1, S2, S1+S2, S3 or S1+S2+S3 in target sheets. ExcelProvider also normalizes S1S2 and S1S2S3. Python uses EScope enum members.
   * - ``s3_category``
     - Category 1–15, or headline.
     - Use 0 or a blank template cell for a headline S3 target. Category emissions may be required for combining targets.
   * - ``coverage_s1``, ``coverage_s2``, ``coverage_s3``
     - Decimal fractions; conditional.
     - Fraction of the relevant scope covered by the target; 70% is 0.7. Fill the scopes the target covers. Scope 3 coverage refers to total scope emissions.
   * - ``reduction_ambition``
     - Decimal fraction; conditional on target type.
     - Reduction ambition; 70% is 0.7. Special T_score targets use temperature fields instead of a reduction-based regression.
   * - ``base_year``, ``end_year``
     - Integer years; required.
     - Target base/end years. Eligibility includes a valid period and a non-expired target end year.
   * - ``start_year``
     - Integer year; optional.
     - Announcement/start year. Falls back to base year when absent.
   * - ``statement_date``
     - Year in ExcelProvider; optional value.
     - For this Excel loader, use a year such as 2024. Its year parser falls back to start_year then base_year. Custom Python target objects accept date values; do not assume full-date Excel values are preserved by this loader.
   * - ``base_year_ghg_s1``, ``base_year_ghg_s2``
     - Numbers in tCO2e; conditional.
     - Base-year emissions supporting relevant scope calculations and splitting combined targets.
   * - ``base_year_ghg_s1s2``, ``base_year_ghg_s3``
     - Numbers in tCO2e; conditional.
     - Combined S1+S2 or S3 base-year emissions where needed by the target calculation.
   * - ``base_year_ts``, ``end_year_ts``
     - Numbers in degrees C; T_score targets.
     - Temperature inputs for the special methodology-defined T_score target type, not a portfolio of ready-made scores.
   * - ``achieved_reduction``
     - Decimal fraction; optional, defaults to 0.
     - Already achieved reduction. A value of 1 or more invalidates the target.
   * - ``target_ids``
     - Text in Excel, list of strings in Python; optional.
     - Identifiers retained to trace source targets. A single Excel string is converted into a one-element list.

Ready-made score input
~~~~~~~~~~~~~~~~~~~~~~

The app requires ``company_id``, ``company_name``, ``investment_value``,
``scope``, ``time_frame`` and ``temperature_score``. Use ``S1S2`` and
``S1S2S3`` for combined scopes and ``SHORT``, ``MID`` or ``LONG`` for
timeframes. Values are case-insensitive in this importer. Each row is
one company/scope/timeframe combination. Add emissions and financial
inputs for the selected aggregation method. See :doc:`DataRequirements`.

CTA input
~~~~~~~~~

CTA column names are ``company_name``, ``isin``, ``lei`` and
``near_term_status``. These identifier names differ from the portfolio's
``company_isin`` and ``company_lei``. See :ref:`sbti-status` for accepted
statuses, matching and storage.

Output data
-----------

Core result fields
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Format
     - Meaning
   * - ``company_id``, ``company_name``
     - Text.
     - Company identity carried through the calculation.
   * - ``scope``
     - EScope / scope code.
     - Scope represented by a result row.
   * - ``time_frame``
     - ETimeFrames / timeframe code.
     - SHORT, MID or LONG, assigned relative to the calculation year with methodology exceptions.
   * - ``temperature_score``
     - Number in degrees C.
     - Company score for a scope/timeframe combination; aggregation outputs contain weighted portfolio scores.
   * - ``temperature_results``
     - Number from 0 to 1.
     - Default-score influence: 0 is target-based, 1 fully default. Combined scopes can be intermediate. The app defaults missing external values to 0; this does not audit score provenance.
   * - ``influence_percentage``
     - Percentage from 0 to 100.
     - Weighted influence of default-score results in an aggregate, distinct from SBTi coverage.
   * - ``sbti_validated``
     - Boolean.
     - Validation flag from the selected data/status enrichment. Interpretation depends on the CTA source and identifiers.
   * - ``annual_reduction_rate``
     - Numeric annual rate.
     - Compound annual reduction used by the scoring formula; the formula converts units/sign for the regression.
   * - ``ar6``
     - Text.
     - Mapped AR6 regression variable.
   * - ``slope``
     - Text.
     - Regression horizon: slopeCA5, slopeCA10 or slopeCA30 in the current configuration.
   * - ``param``, ``intercept``
     - Numbers.
     - Regression coefficient and intercept. They are not restricted to fractions between 0 and 1.
   * - ``target_ids``
     - List of strings.
     - Source targets contributing to the result.
   * - ``contribution``, ``contribution_relative``
     - Numbers in aggregated output.
     - Company contribution in score units and relative percentage contribution to the aggregate.

Additional input and intermediate columns may appear in the company-score
DataFrame. The complete Python structures are described in
:doc:`autoapi/index`. Use :doc:`FunctionalOverview` for interpretation,
and :doc:`using_app` for the actual app downloads.
