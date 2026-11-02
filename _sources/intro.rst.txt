Introduction to temperature scoring and portfolio coverage methods
==================================================================

Temperature scoring translates companies' public emissions-reduction
targets into temperature-alignment scores and combines them at portfolio
level. Portfolio coverage measures the weighted share of holdings with
SBTi-validated near-term targets. The two outputs are complementary:
ambitious disclosed targets and independently validated targets are not
the same thing.

The WWF tool implements the
`CDP-WWF Temperature Scoring Methodology v1.5 (September 2024)
<https://wwfint.awsassets.panda.org/downloads/cdp-wwf-temperature-scoring-methodology---september-2024.pdf>`_.
The underlying project originated in the SBTi-Finance tool. Older SBTi
methodology documents describe that history; the WWF method above is the
reference for this implementation. A computed score does not in itself
constitute official approval of a financial institution's target.
For target submissions, consult the
`SBTi's financial institution resources <https://sciencebasedtargets.org/financial-institutions>`_
for the applicable criteria.

Why have we built this tool?
----------------------------

The tool makes target-based portfolio analysis transparent and repeatable.
It helps users identify the companies and sectors that drive a portfolio
score, compare weighting choices, and explore engagement assumptions.
The code and methodology can be inspected, and detailed outputs can be
exported for further analysis.

Why have we built the WWF Finance tool in this way?
---------------------------------------------------

The calculation engine is an open-source Python package that can be
integrated into existing workflows and infrastructure. It uses
user-supplied data rather than requiring a particular commercial provider.
The WWF version uses regressions based on IPCC AR6 scenario data.

The Streamlit app provides a graphical interface, Docker packages the app
for local execution, and notebooks offer step-by-step explanations.
Choose the environment appropriate for your data: the public demo and
Colab process data on external infrastructure, whereas local installations
run calculations on your computer. See :doc:`index` for the entry points.

What can I use the WWF Finance tool for?
----------------------------------------

* Calculate company and portfolio temperature scores by scope and timeframe.
* Compare sector, region or other groupings using Python/notebooks and
  inspect companies and hotspots in the app.
* Aggregate ready-made company scores provided by a data supplier.
* Measure investment-weighted coverage by SBTi-validated near-term targets.
* Explore how changed target assumptions could alter a portfolio score.
* Export results and underlying data for review and reporting.

Scores describe the ambition of disclosed targets under the method's
assumptions, not a prediction or guarantee of realized warming. Consider
data quality, missing information, default scores and the selected
aggregation method when interpreting them.

What data do I need to use the tool?
------------------------------------

For target-based scoring, supply a portfolio, company fundamentals and
emissions, and company target data. For ready-made score aggregation,
supply company scores and the required weighting data. For standalone
WATS coverage, supply a portfolio with identifiers and investment values
plus CTA data. See :doc:`DataRequirements` for templates and formats.

Where can I find the data?
--------------------------

Use company disclosures, your own data collection or a data provider
whose information can be mapped to the templates. The sample files are
illustrative and should not be treated as real investment data.
SBTi status is obtained from CTA data as described in :ref:`sbti-status`.
Coverage depends on identifier quality and the selected CTA snapshot.

Overview of how the tool works
------------------------------

1. **Benchmarks.** Regression models based on IPCC AR6 scenarios map
   target ambition to temperature scores. These parameters are supplied
   with the package; ordinary users do not need to rebuild them. See
   `ITR-regression <https://github.com/WWF-Sweden/ITR-regression>`_ for the
   benchmark project.
2. **Company assessment.** Targets are validated and selected by the
   method's hierarchy. Compound annual reduction (CAR), target type,
   sector, scope and timeframe determine the applicable score. Combined
   scopes use emissions data. A default score applies when no eligible
   score can be calculated; the standard configuration uses 3.4°C with
   a 1.5°C temperature floor.
3. **Portfolio scoring.** Company scores are combined using the selected
   weighting method. Contributions identify which holdings drive the
   aggregate. See :doc:`FunctionalOverview` for the available methods.

Portfolio coverage is a separate status-and-weighting calculation; it
does not require calculating company temperature scores first.

What-if analyses explore changed target assumptions after a baseline run.
The scenario caps are 1.75°C or 1.5°C according to the chosen engagement
type. Selective engagement uses ``engagement_target`` in Python, or the
app's candidate selector. These simulations do not change observed SBTi
validation status.

What are the outputs the tool generates?
----------------------------------------

Company results contain temperature scores by timeframe and scope with
supporting data. Portfolio results contain weighted scores, contributions
and information about default-score influence. Grouped results can be
requested in Python. Coverage is a percentage from 0 to 100.

The app provides portfolio, hotspot, company and scenario views and
CSV/Excel downloads. Python and notebooks provide more control over
grouping, detailed data and anonymization. See :doc:`using_app` for the
actual app exports and :doc:`notebooks` for worked reporting examples.
