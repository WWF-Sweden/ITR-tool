# TODO

## RuntimeWarning in `portfolio_aggregation.py`

**File:** `ITR/portfolio_aggregation.py`

`_check_column` only validates `NaN`/null values (`pd.isnull()`), not zero values.
Companies with `0` in denominator columns (`market_cap`, `company_total_assets`, `company_revenue`, etc.) pass validation but trigger a numpy `RuntimeWarning: divide by zero encountered in true_divide` during the vectorized division in `_calculate_aggregate_score`.

The existing `try/except ZeroDivisionError` blocks don't catch this — pandas returns `inf` silently rather than raising a Python exception, so those blocks are effectively dead code for this case.

**Fix options:**
- Extend `_check_column` to also reject zero values in denominator columns.
- Wrap the division operations with `np.errstate(divide='ignore', invalid='ignore')` and handle `inf`/`NaN` explicitly afterwards.
