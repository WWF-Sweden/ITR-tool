import copy
import datetime
import unittest
from typing import List
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ITR
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
    ScenarioType,
    TemperatureScore,
)
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.data.data_provider import DataProvider


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


class EndToEndTest(unittest.TestCase):
    """
    This class is containing a set of end-to-end tests:
    - basic flow from creating companies/targets up to calculating aggregated values
    - edge cases for scenarios and grouping
    - high load tests (>1000 targets)
    - testing of all different input values and running through the whole process (tbd)
    """

    def setUp(self):
        company_id = "BaseCompany"
        self.BASE_COMP_SCORE = 1.5
        self.BASE_COMP_SCORE_GROUP = 1.5  # Will be updated if needed

        # target end years which align to (short, mid, long) time frames
        # this is a goo idea but needs reflected in refrence scores
        self.short_end_year = datetime.datetime.now().year + 2
        self.mid_end_year = datetime.datetime.now().year + 5
        self.long_end_year = datetime.datetime.now().year + 25

        self.company_base = IDataProviderCompany(
            company_name=company_id,
            company_id=company_id,
            ghg_s1=100,
            ghg_s2=100,
            ghg_s1s2=200,
            ghg_s3=0,
            company_revenue=100,
            company_market_cap=100,
            company_enterprise_value=100,
            company_total_assets=100,
            company_cash_equivalents=100,
            isic="A12",
            country="Unknown",
            region="Unknown",
            sector="Unknown",
            industry_level_1="Unknown",
            industry_level_2="Unknown",
            industry_level_3="Unknown",
            industry_level_4="Unknown",
            sbti_validated=False,
        )
        # define targets
        self.target_base = self._create_target_with_defaults(company_id)

        # pf
        self.pf_base = PortfolioCompany(
            company_name=company_id,
            company_id=company_id,
            investment_value=100,
            company_isin=company_id,
            company_lei=company_id,
        )

    def _create_target_with_defaults(self, company_id: str, **kwargs) -> IDataProviderTarget:
        """
        calls IDataProviderTarget constructor with defaults
        can override specific params with kwargs
        """
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
            end_year=self.mid_end_year,
            target_ids="target_base"
        )
        defaults.update(kwargs)
        return IDataProviderTarget(**defaults) # type: ignore

    def test_basic(self):
        """
        This test is just a very basic workflow going through all calculations up to temp score
        """

        # Setup test provider
        company = copy.deepcopy(self.company_base)
        target = self._create_target_with_defaults(company_id=company.company_id)
        data_provider = TestDataProvider(companies=[company], targets=[target])

        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID, ETimeFrames.SHORT, ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        # portfolio data
        pf_company = copy.deepcopy(self.pf_base)
        portfolio_data = ITR.utils.get_data([data_provider], [pf_company])

        # Verify data
        scores = temp_score.calculate(portfolio_data)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores.index), 3)

    def test_default_score(self):
        """
        test default score assignment
        """
        # Setup test provider
        company = copy.deepcopy(self.company_base)
        target = self._create_target_with_defaults(company_id=company.company_id)
        data_provider = TestDataProvider(companies=[company], targets=[target])

        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID, ETimeFrames.SHORT, ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        # portfolio data
        pf_company = copy.deepcopy(self.pf_base)
        portfolio_data = ITR.utils.get_data([data_provider], [pf_company])

        # Verify data
        scores = temp_score.calculate(portfolio_data)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores.index), 3)

    def test_chaos(self):
        # TODO: go thru lots of different parameters on company & target level and try to break it
        pass

    def test_target_grouping(self):
        """
        This test is checking the target grouping in the target validation from begin to end.
        """

        companies, targets, pf_companies = self.create_base_companies(
            ["A", "B", "C", "D"]
        )
        target = self._create_target_with_defaults(
            company_id="A",
            coverage_s1=0.75,
            coverage_s2=0.75,
            coverage_s3=0.75,
        )

        targets.append(target)
        target = self._create_target_with_defaults(
            company_id="A",
            coverage_s1=0.99,
            coverage_s2=0.99,
            coverage_s3=0.99,
        )
        targets.append(target)

        target = self._create_target_with_defaults(
            company_id="B",
            scope=EScope.S3,
            coverage_s1=0.75,
            coverage_s2=0.75,
            coverage_s3=0.49,
        )
        targets.append(target)

        target = self._create_target_with_defaults(
            company_id="B",
            scope=EScope.S3,
            coverage_s1=0.99,
            coverage_s2=0.99,
            coverage_s3=0.49,
            end_year=2035,
        )
        targets.append(target)

        target = self._create_target_with_defaults(
            company_id="D",
            coverage_s1=0.95,
            coverage_s2=0.95,
            target_type="int",
            intensity_metric="Revenue",
        )
        targets.append(target)

        data_provider = TestDataProvider(companies=companies, targets=targets)

        # Calculate scores & Aggregated values
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2, EScope.S1S2S3],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([data_provider], pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify that results exist
        self.assertAlmostEqual(
            agg_scores.mid.S1S2.all.score, self.BASE_COMP_SCORE_GROUP, places=4
        )

    def test_target_ids(self):
        """
        Test handling of target_ids through the v1.5 pipeline:
            - S1S2 targets get split into S1+S2 with '_1'/'_2' suffixes
            - S1S2S3 targets get split into S1S2+S3, then S1S2 further split
            - Targets outside requested time frames are not used
            - Higher coverage targets are preferred
        """
        # given
        companies, targets, pf_companies = self.create_base_companies(
            ["A", "B", "C"]
        )

        # Company A: add separate S1 and S2 targets with higher coverage for MID
        target_a_s1 = self._create_target_with_defaults(
            company_id="A",
            scope=EScope.S1,
            end_year=self.mid_end_year,
            coverage_s1=1.0,
            target_ids=["A_s1_high_cov"],
        )
        targets.append(target_a_s1)

        target_a_s2 = self._create_target_with_defaults(
            company_id="A",
            scope=EScope.S2,
            end_year=self.mid_end_year,
            coverage_s2=0.99,
            target_ids=["A_s2_high_cov"],
        )
        targets.append(target_a_s2)

        # Company A: SHORT target should be excluded from MID/LONG results
        target_a_short = self._create_target_with_defaults(
            company_id="A",
            scope=EScope.S2,
            end_year=self.short_end_year,
            coverage_s2=1.0,
            target_ids=["A_short_only"],
        )
        targets.append(target_a_short)

        # Company B: S1S2S3 target for LONG - should be split
        target_b_s1s2s3 = self._create_target_with_defaults(
            company_id="B",
            scope=EScope.S1S2S3,
            end_year=self.long_end_year,
            coverage_s3=0.8,
            base_year_ghg_s3=50,
            target_ids=["B_s1s2s3_long"],
        )
        targets.append(target_b_s1s2s3)

        # when
        data_provider = TestDataProvider(companies=companies, targets=targets)
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID, ETimeFrames.LONG],
            scopes=[EScope.S1S2, EScope.S1S2S3],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )
        portfolio_data = ITR.utils.get_data([data_provider], pf_companies)
        scores = temp_score.calculate(portfolio_data)

        # then
        used_ids = set(
            tid for tids in scores["target_ids"].tolist()
            for tid in (tids or [])
        )
        # Strip _1/_2 suffixes to get base target IDs
        base_ids = set()
        for tid in used_ids:
            if tid.endswith("_1") or tid.endswith("_2"):
                base_ids.add(tid[:-2])
            else:
                base_ids.add(tid)

        # Company A's high-coverage S1+S2 targets should be used for MID
        self.assertIn("A_s1_high_cov", base_ids,
                       "High-coverage S1 target should be used")
        self.assertIn("A_s2_high_cov", base_ids,
                       "High-coverage S2 target should be used")

        # SHORT-only target should not appear in MID/LONG results
        self.assertNotIn("A_short_only", base_ids,
                          "SHORT target should not appear in MID/LONG results")

        # Company B's S1S2S3 target should be split and used for LONG
        self.assertIn("B_s1s2s3_long", base_ids,
                       "S1S2S3 target should be split and used")

    def test_basic_flow(self):
        """
        This test is going all the way to the aggregated calculations
        """

        companies, targets, pf_companies = self.create_base_companies(["A", "B"])

        data_provider = TestDataProvider(companies=companies, targets=targets)

        # Calculate scores & Aggregated values
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2, EScope.S1S2S3],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([data_provider], pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify that results exist
        self.assertEqual(agg_scores.mid.S1S2.all.score, self.BASE_COMP_SCORE)

    # Run some regression tests
    # @unittest.skip("only run for longer test runs")
    def test_regression_companies(self):
        nr_companies = 1000

        # test 10000 companies
        companies: List[IDataProviderCompany] = []
        targets: List[IDataProviderTarget] = []
        pf_companies: List[PortfolioCompany] = []

        for i in range(nr_companies):
            company_id = f"Company {str(i)}"
            # company
            company = copy.deepcopy(self.company_base)
            company.company_id = company_id
            companies.append(company)

            # target
            target = copy.deepcopy(self.target_base)
            target.company_id = company_id
            targets.append(target)

            # pf company
            pf_company = PortfolioCompany(
                company_name=company_id,
                company_id=company_id,
                investment_value=100,
                company_isin=company_id,
                company_lei=company_id
            )
            pf_companies.append(pf_company)

        data_provider = TestDataProvider(companies=companies, targets=targets)

        # Calculate scores & Aggregated values
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data([data_provider], pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        self.assertAlmostEqual(agg_scores.mid.S1S2.all.score, self.BASE_COMP_SCORE)

    def test_grouping(self):
        """
        Testing the grouping feature with two different industry levels and making sure the results are present
        """
        # make 2+ companies and group them together
        industry_levels = ["Manufacturer", "Energy"]
        company_ids = ["A", "B"]
        companies_all: List[IDataProviderCompany] = []
        targets_all: List[IDataProviderTarget] = []
        pf_companies_all: List[PortfolioCompany] = []

        for ind_level in industry_levels:
            company_ids_with_level = [
                f"{ind_level}_{company_id}" for company_id in company_ids
            ]

            companies, targets, pf_companies = self.create_base_companies(
                company_ids_with_level
            )
            for company in companies:
                company.industry_level_1 = ind_level

            companies_all.extend(companies)
            targets_all.extend(targets)
            pf_companies_all.extend(pf_companies)

        data_provider = TestDataProvider(companies=companies_all, targets=targets_all)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
            grouping=["industry_level_1"],
        )

        portfolio_data = ITR.utils.get_data([data_provider], pf_companies_all)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        for ind_level in industry_levels:
            self.assertAlmostEqual(
                agg_scores.mid.S1S2.grouped[ind_level].score, self.BASE_COMP_SCORE
            )

    def test_score_cap(self):
        companies, targets, pf_companies = self.create_base_companies(["A"])
        data_provider = TestDataProvider(companies=companies, targets=targets)

        # add a Scenario that will trigger the score cap function
        scenario = Scenario()
        scenario.engagement_type = EngagementType.SET_TARGETS
        scenario.scenario_type = ScenarioType.APPROVED_TARGETS

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.MID],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
            scenario=scenario,
        )

        portfolio_data = ITR.utils.get_data([data_provider], pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # add verification

    def create_base_companies(self, company_ids: List[str]):
        """
        This is a helper method to create base companies that can be used for the test cases
        """
        companies: List[IDataProviderCompany] = []
        targets: List[IDataProviderTarget] = []
        pf_companies: List[PortfolioCompany] = []
        for company_id in company_ids:
            # company
            company = copy.deepcopy(self.company_base)
            company.company_id = company_id
            companies.append(company)

            # pf company
            pf_company = PortfolioCompany(
                company_name=company_id,
                company_id=company_id,
                investment_value=100,
                company_isin=company_id,
                company_lei=company_id
            )

            target = self._create_target_with_defaults(company_id=company_id)

            pf_companies.append(pf_company)
            targets.append(target)

        return companies, targets, pf_companies


if __name__ == "__main__":
    test = EndToEndTest()
    test.setUp()
    test.test_basic()
    test.test_basic_flow()
    test.test_regression_companies()
    test.test_score_cap()
    test.test_target_grouping()