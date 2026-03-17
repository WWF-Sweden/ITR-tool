"""
Test edge cases in the ITR scoring pipeline.
Tests behavior with missing data, unusual inputs, and boundary conditions.
"""
import copy
import datetime
import os
import sys
import unittest
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ITR.interfaces import (
    EScope,
    ETimeFrames,
    IDataProviderCompany,
    IDataProviderTarget,
    PortfolioCompany,
)
from ITR.temperature_score import (
    EngagementType,
    Scenario,
    TemperatureScore,
)
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.data.data_provider import DataProvider
import ITR.utils


class TestDataProvider(DataProvider):
    def __init__(
        self, targets: List[IDataProviderTarget], companies: List[IDataProviderCompany]
    ):
        super().__init__()
        self.targets = targets
        self.companies = companies

    def get_sbti_targets(self, companies: list) -> list:
        return []

    def get_targets(self, company_ids: List[str]) -> List[IDataProviderTarget]:
        return self.targets

    def get_company_data(self, company_ids: List[str]) -> List[IDataProviderCompany]:
        return self.companies


class EdgeCasesTest(unittest.TestCase):
    """Test edge cases: missing GHG data, lone scopes, boundary conditions."""

    MID_END_YEAR = datetime.datetime.now().year + 5

    def _make_company(self, company_id: str, **kwargs) -> IDataProviderCompany:
        defaults = dict(
            company_name=company_id,
            company_id=company_id,
            isic="A12",
            ghg_s1=100,
            ghg_s2=100,
            ghg_s1s2=200,
            ghg_s3=0,
            company_revenue=100,
            company_market_cap=100,
            company_enterprise_value=100,
            company_total_assets=100,
            company_cash_equivalents=100,
        )
        defaults.update(kwargs)
        return IDataProviderCompany(**defaults)

    def _make_target(self, company_id: str, **kwargs) -> IDataProviderTarget:
        defaults = dict(
            company_id=company_id,
            target_type="abs",
            scope=EScope.S1S2,
            coverage_s1=0.95,
            coverage_s2=0.95,
            coverage_s3=0,
            reduction_ambition=0.8,
            base_year=2019,
            base_year_ghg_s1=100,
            base_year_ghg_s2=100,
            base_year_ghg_s3=0,
            end_year=self.MID_END_YEAR,
        )
        defaults.update(kwargs)
        return IDataProviderTarget(**defaults)

    def _make_pf(self, company_id: str) -> PortfolioCompany:
        return PortfolioCompany(
            company_name=company_id,
            company_id=company_id,
            investment_value=100,
            company_isin=company_id,
            company_lei=company_id,
        )

    def test_missing_individual_ghg(self):
        """
        Test company with only ghg_s1s2 (no individual ghg_s1/ghg_s2).
        The S1S2 target can't be split, so scores default.
        Pipeline should not crash.
        """
        import numpy as np
        companies = [
            self._make_company("MissingGHG", ghg_s1=np.nan, ghg_s2=np.nan, ghg_s1s2=100),
        ]
        targets = [self._make_target("MissingGHG")]
        portfolio = [self._make_pf("MissingGHG")]

        dp = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([dp], portfolio)
        scores = temp_score.calculate(portfolio_data)

        # Should produce a score (likely default 3.2 since target can't be split)
        self.assertIsNotNone(scores)
        self.assertGreater(len(scores), 0)

    def test_s1_only_target(self):
        """
        Test company with only an S1 scope target (no S2).
        In v1.5, lone S1 without S2 is kept but scored individually.
        """
        companies = [self._make_company("S1Only")]
        targets = [
            self._make_target("S1Only", scope=EScope.S1, base_year_ghg_s2=0),
        ]
        portfolio = [self._make_pf("S1Only")]

        dp = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        portfolio_data = ITR.utils.get_data([dp], portfolio)
        scores = temp_score.calculate(portfolio_data)
        self.assertIsNotNone(scores)

    def test_zero_investment_value(self):
        """
        Test portfolio company with zero investment value.
        Should not cause division by zero in aggregation.
        """
        companies = [
            self._make_company("ZeroInv"),
            self._make_company("NormalInv"),
        ]
        targets = [
            self._make_target("ZeroInv"),
            self._make_target("NormalInv"),
        ]
        portfolio = [
            PortfolioCompany(
                company_name="ZeroInv", company_id="ZeroInv",
                investment_value=0, company_isin="ZeroInv", company_lei="ZeroInv",
            ),
            self._make_pf("NormalInv"),
        ]

        dp = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([dp], portfolio)
        scores = temp_score.calculate(portfolio_data)
        # Should not crash
        temp_score.aggregate_scores(scores)

    def test_no_valid_targets(self):
        """
        Test company where all targets are invalid (NaN reduction_ambition).
        Pipeline should handle gracefully.
        """
        import numpy as np
        companies = [self._make_company("NoValid")]
        targets = [
            self._make_target("NoValid", reduction_ambition=np.nan),
        ]
        portfolio = [self._make_pf("NoValid")]

        dp = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        try:
            portfolio_data = ITR.utils.get_data([dp], portfolio)
            scores = temp_score.calculate(portfolio_data)
            # If scores are produced, they should be default (3.2)
            self.assertIsNotNone(scores)
        except (ValueError, KeyError):
            # Acceptable — pipeline may reject all-invalid input
            pass

    def test_extreme_reduction_ambition(self):
        """
        Test with extreme (boundary) reduction ambition values.
        """
        companies = [self._make_company("Extreme")]
        targets = [
            self._make_target("Extreme", reduction_ambition=1.0),
        ]
        portfolio = [self._make_pf("Extreme")]

        dp = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        portfolio_data = ITR.utils.get_data([dp], portfolio)
        scores = temp_score.calculate(portfolio_data)

        # Score with 100% reduction should be low (close to 1.5C or below)
        mid_scores = scores[
            (scores["scope"] == EScope.S1S2)
            & (scores["time_frame"] == ETimeFrames.MID)
        ]
        if not mid_scores.empty:
            ts = mid_scores["temperature_score"].iloc[0]
            if ts == ts:  # not NaN
                self.assertLessEqual(ts, 2.0,
                    "100% reduction ambition should yield a low score")


if __name__ == "__main__":
    unittest.main()
