Functional Overview
===================

Temperature Score
-----------------

The engine translates eligible company targets into temperature scores
using the `CDP-WWF methodology v1.5
<https://wwfint.awsassets.panda.org/downloads/cdp-wwf-temperature-scoring-methodology---september-2024.pdf>`_
and packaged AR6 regression parameters. It then aggregates company scores
using portfolio weights. Companies can receive default scores when no
eligible target-based score is available for a combination.
The default configuration uses 3.4°C and a floor of 1.5°C.

Time Frames
~~~~~~~~~~~

Select ``SHORT``, ``MID`` or ``LONG``. The standard target assignment is
forward-looking from the calculation year: less than five years to the
target end year is short, five through ten is mid, and more than ten is
long, subject to the method's target eligibility and exceptions.
The ``T_score`` target type has a five-year adjustment in timeframe
assignment. Sample results can therefore change over time.
The app initially selects MID; Python callers explicitly supply timeframes.

Scopes
~~~~~~

The scopes are S1, S2, S1+S2, S3 and S1+S2+S3. In Python and ready-made
score files, the combined enum names are ``S1S2`` and ``S1S2S3``.
Combined scores use emissions data. The app adds prerequisite scopes
internally; Python callers should include them, as demonstrated in
:doc:`getting_started`.

Aggregation methods
~~~~~~~~~~~~~~~~~~~

* **WATS**: weighted average using portfolio investment values.
* **TETS**: total emissions weighted temperature score.
* **MOTS**: owned emissions weighted using market capitalization.
* **EOTS**: owned emissions weighted using enterprise value.
* **ECOTS**: owned emissions weighted using enterprise value plus cash.
* **AOTS**: owned emissions weighted using total assets.
* **ROTS**: owned emissions weighted using revenue.

The weighting method changes the question answered by the portfolio
score. Required emissions and financial values are listed in
:doc:`DataRequirements`. Company scores can also be examined before
aggregation. Existing provider scores can be aggregated without
recalculating targets; see notebook 4b in :doc:`notebooks`.

Grouping data
~~~~~~~~~~~~~

In Python, pass ``grouping=["sector"]`` or a list of grouping columns
to obtain weighted scores and contributions for those groups. Categories
can include region, country, sector, industry levels and user fields
supplied through the portfolio. See notebook 1 in :doc:`notebooks`.
The app offers hotspot and company views with its own filters and chart
controls; it does not expose every Python grouping option. Read each
chart's labels to distinguish descriptive statistics from weighted scores.

Choose fields to show
~~~~~~~~~~~~~~~~~~~~~

The company-score DataFrame includes company, timeframe, scope and score
with supporting input and calculation fields. Python lets you select
columns, inspect the detailed data and create custom exports. The app's
downloads are described in :doc:`using_app`; they are not a complete
archive of every input and intermediate result.

What-If Analyses
~~~~~~~~~~~~~~~~

Scenarios change assumptions to explore engagement effects; they do not
prove actual target improvements or change observed SBTi validation.
The Python scenarios are:

* ``TARGETS`` (1): use 1.75°C in place of the default score.
* ``APPROVED_TARGETS`` (2): cap scores with target references, using
  1.5°C when configured for the SBTi-target engagement type.
* ``HIGHEST_CONTRIBUTORS`` (3): cap the ten highest contributors for
  each selected scope/timeframe combination.
* ``HIGHEST_CONTRIBUTORS_APPROVED`` (4): cap companies whose
  ``engagement_target`` is true. Despite its enum name, this is the
  selective-engagement scenario in the implementation.

``SET_TARGETS`` applies a 1.75°C cap and ``SET_SBTI_TARGETS`` a 1.5°C cap
where engagement caps are used. Explicitly set both the scenario and
engagement type in Python. See notebook 3 for worked combinations.
The app offers selectors and an engagement-candidate workflow. Scenarios
are unavailable in the app's pre-scored data mode.

Portfolio coverage
------------------

Coverage is the weighted percentage of holdings with validated near-term
SBTi targets in the CTA data used for the run. With WATS, the numerator
is investment value with matched validated targets and the denominator
is total investment value. Unmatched holdings contribute zero to the
numerator. The result is from 0 to 100, not a fraction requiring another
multiplication by 100.

Portfolio-only WATS coverage does not require a temperature calculation.
The standalone notebook uses pandas without ``wwf-itr``. The Python
package also provides ``PortfolioCoverageTVP`` for this purpose. Other
weighting methods require their additional company data.
For CTA formats and matching limitations, see :ref:`sbti-status`.

Output options
--------------

Python can return company scores, aggregate scores, grouped scores and
company contributions with supporting data. ``temperature_results``
tracks default-score influence: 0 indicates a target-based score and
1 a fully default score; combined scopes can be intermediate. The
aggregated ``influence_percentage`` expresses default-score influence
as a percentage. This is distinct from SBTi portfolio coverage.

For ready-made scores, the app supplies ``temperature_results=0`` when
the field is absent. Such an influence value does not independently
verify the provenance or quality of external scores.

The app provides CSV and Excel exports. Python/notebooks offer custom
selection and anonymization. Check all exported identifiers and other
potentially identifying fields before sharing data; see notebook 5 for
reporting examples.
