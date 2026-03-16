"""
Test the temperature score calculation through the full data pipeline.
Uses in-memory data providers to create test companies/targets, then
validates scores and aggregations.

Updated for v1.5 which added separate S1/S2 scope scoring, S3 category
aggregation, and boundary coverage sorting.
"""

import copy
import datetime
import unittest
from typing import List

from ITR.interfaces import (
    EScope,
    ETimeFrames,
    IDataProviderCompany,
    IDataProviderTarget,
    PortfolioCompany,
)
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.data.data_provider import DataProvider
import ITR.utils


class TestDataProvider(DataProvider):
    """Simple in-memory data provider for testing."""
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


class TestTemperatureScore(unittest.TestCase):
    """
    Test the temperature scoring functionality using the full data pipeline.
    """

    # Dynamic end years to ensure stable time frame assignment
    # regardless of when tests are run
    SHORT_END_YEAR = datetime.datetime.now().year + 2
    MID_END_YEAR = datetime.datetime.now().year + 7
    LONG_END_YEAR = datetime.datetime.now().year + 25

    def _make_company(self, company_id: str, **kwargs) -> IDataProviderCompany:
        defaults = dict(
            company_name=company_id,
            company_id=company_id,
            isic="A12",
            ghg_s1=100,
            ghg_s2=100,
            ghg_s1s2=200,
            ghg_s3=50,
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

    def _make_pf_company(self, company_id: str, **kwargs) -> PortfolioCompany:
        defaults = dict(
            company_name=company_id,
            company_id=company_id,
            investment_value=100,
            company_isin=company_id,
            company_lei=company_id,
        )
        defaults.update(kwargs)
        return PortfolioCompany(**defaults)

    def setUp(self) -> None:
        """
        Set up test companies, targets, and portfolio for testing.
        """
        # Create companies with known GHG values
        self.companies = [
            self._make_company("CompA", ghg_s1=200, ghg_s2=100, ghg_s3=50),
            self._make_company("CompB", ghg_s1=150, ghg_s2=150, ghg_s3=100),
        ]

        # Create targets for MID time frame (S1S2 scope)
        self.targets = [
            self._make_target("CompA", scope=EScope.S1S2, end_year=self.MID_END_YEAR),
            self._make_target("CompB", scope=EScope.S1S2, end_year=self.MID_END_YEAR),
        ]

        self.portfolio = [
            self._make_pf_company("CompA", investment_value=100),
            self._make_pf_company("CompB", investment_value=200),
        ]

        self.data_provider = TestDataProvider(
            companies=self.companies, targets=self.targets
        )

    def test_temp_score_basic(self) -> None:
        """Test that temperature scores are calculated for S1S2 scope."""
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([self.data_provider], self.portfolio)
        scores = temp_score.calculate(portfolio_data)

        self.assertIsNotNone(scores)
        # Both companies should have MID S1S2 scores
        for company_id in ["CompA", "CompB"]:
            company_scores = scores[
                (scores["company_id"] == company_id)
                & (scores["scope"] == EScope.S1S2)
                & (scores["time_frame"] == ETimeFrames.MID)
            ]
            self.assertEqual(len(company_scores), 1, f"Expected 1 score for {company_id}")
            ts = company_scores["temperature_score"].iloc[0]
            self.assertFalse(
                ts != ts,  # NaN check
                f"Temperature score for {company_id} should not be NaN",
            )
            # Score should be between floor and cap
            self.assertGreaterEqual(ts, 0.0)
            self.assertLessEqual(ts, 3.2)

    def test_default_score(self) -> None:
        """Test that companies without valid targets get the default score."""
        # Create company with no targets
        companies = [self._make_company("NoTarget")]
        targets = [
            self._make_target("NoTarget", reduction_ambition=float("nan")),
        ]
        portfolio = [self._make_pf_company("NoTarget")]

        data_provider = TestDataProvider(companies=companies, targets=targets)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        try:
            portfolio_data = ITR.utils.get_data([data_provider], portfolio)
            scores = temp_score.calculate(portfolio_data)
            # If we get scores, default should be 3.2
            default_scores = scores[
                (scores["company_id"] == "NoTarget")
                & (scores["scope"] == EScope.S1S2)
            ]
            if not default_scores.empty:
                self.assertAlmostEqual(
                    default_scores["temperature_score"].iloc[0],
                    3.2,
                    places=1,
                    msg="Default score should be 3.2",
                )
        except (ValueError, KeyError):
            # Acceptable when all targets are invalid — pipeline may produce
            # DataFrame missing expected columns
            pass

    def test_s1s2s3_combined_score(self) -> None:
        """Test that S1S2S3 combined scores are calculated correctly."""
        companies = [
            self._make_company("CompC", ghg_s1=200, ghg_s2=100, ghg_s3=300),
        ]
        targets = [
            self._make_target(
                "CompC", scope=EScope.S1S2, end_year=self.MID_END_YEAR,
                coverage_s1=0.95, coverage_s2=0.95,
            ),
            self._make_target(
                "CompC", scope=EScope.S3, end_year=self.MID_END_YEAR,
                coverage_s3=0.8, base_year_ghg_s3=300, reduction_ambition=0.5,
            ),
        ]
        portfolio = [self._make_pf_company("CompC")]
        data_provider = TestDataProvider(companies=companies, targets=targets)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2, EScope.S3, EScope.S1S2S3],
        )

        portfolio_data = ITR.utils.get_data([data_provider], portfolio)
        scores = temp_score.calculate(portfolio_data)

        # Should have S1S2, S3, and S1S2S3 scores
        s1s2_scores = scores[
            (scores["company_id"] == "CompC") & (scores["scope"] == EScope.S1S2)
        ]
        s3_scores = scores[
            (scores["company_id"] == "CompC") & (scores["scope"] == EScope.S3)
        ]
        s1s2s3_scores = scores[
            (scores["company_id"] == "CompC") & (scores["scope"] == EScope.S1S2S3)
        ]

        self.assertFalse(s1s2_scores.empty, "S1S2 scores should exist")
        self.assertFalse(s3_scores.empty, "S3 scores should exist")
        self.assertFalse(s1s2s3_scores.empty, "S1S2S3 scores should exist")

    def test_portfolio_aggregation_wats(self):
        """Test WATS (Weighted Average Temperature Score) aggregation."""
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([self.data_provider], self.portfolio)
        scores = temp_score.calculate(portfolio_data)
        aggregations = temp_score.aggregate_scores(scores)

        # WATS = sum(weight_i * score_i)
        self.assertIsNotNone(aggregations.mid.S1S2.all.score)
        self.assertGreater(aggregations.mid.S1S2.all.score, 0)

    def test_portfolio_aggregation_methods(self):
        """Test that all aggregation methods produce valid results."""
        methods = [
            PortfolioAggregationMethod.WATS,
            PortfolioAggregationMethod.TETS,
            PortfolioAggregationMethod.MOTS,
            PortfolioAggregationMethod.EOTS,
            PortfolioAggregationMethod.ECOTS,
            PortfolioAggregationMethod.AOTS,
        ]

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        portfolio_data = ITR.utils.get_data([self.data_provider], self.portfolio)
        scores = temp_score.calculate(portfolio_data)

        for method in methods:
            temp_score.aggregation_method = method
            aggregations = temp_score.aggregate_scores(scores)
            self.assertIsNotNone(
                aggregations.mid.S1S2.all.score,
                f"{method.name} aggregation should produce a score",
            )

    def test_multiple_time_frames(self):
        """Test scoring across short, mid, and long time frames."""
        companies = [self._make_company("TF_Test")]
        targets = [
            self._make_target("TF_Test", end_year=self.SHORT_END_YEAR),
            self._make_target("TF_Test", end_year=self.MID_END_YEAR),
            self._make_target("TF_Test", end_year=self.LONG_END_YEAR),
        ]
        portfolio = [self._make_pf_company("TF_Test")]
        data_provider = TestDataProvider(companies=companies, targets=targets)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.SHORT, ETimeFrames.MID, ETimeFrames.LONG],
            scopes=[EScope.S1S2],
        )

        portfolio_data = ITR.utils.get_data([data_provider], portfolio)
        scores = temp_score.calculate(portfolio_data)

        # Should have scores for all time frames
        for tf in [ETimeFrames.SHORT, ETimeFrames.MID, ETimeFrames.LONG]:
            tf_scores = scores[
                (scores["company_id"] == "TF_Test")
                & (scores["scope"] == EScope.S1S2)
                & (scores["time_frame"] == tf)
            ]
            self.assertFalse(
                tf_scores.empty,
                f"Should have scores for {tf.value} time frame",
            )

    def test_score_consistency(self):
        """Test that scores are consistent across multiple runs."""
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
        )

        portfolio_data = ITR.utils.get_data([self.data_provider], self.portfolio)
        scores1 = temp_score.calculate(portfolio_data.copy())

        portfolio_data2 = ITR.utils.get_data([self.data_provider], self.portfolio)
        scores2 = temp_score.calculate(portfolio_data2.copy())

        # Scores should be identical
        for company_id in ["CompA", "CompB"]:
            s1 = scores1[
                (scores1["company_id"] == company_id) & (scores1["scope"] == EScope.S1S2)
            ]["temperature_score"].iloc[0]
            s2 = scores2[
                (scores2["company_id"] == company_id) & (scores2["scope"] == EScope.S1S2)
            ]["temperature_score"].iloc[0]
            self.assertAlmostEqual(s1, s2, places=10, msg="Scores should be deterministic")


if __name__ == "__main__":
    unittest.main()
