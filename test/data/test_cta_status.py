import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ITR.configs import PortfolioCoverageTVPConfig
from ITR.data.sbti import SBTi
from ITR.interfaces import IDataProviderCompany
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.portfolio_coverage_tvp import PortfolioCoverageTVP


class TestCTAStatus(unittest.TestCase):
    def setUp(self):
        class Config(PortfolioCoverageTVPConfig):
            USE_LOCAL_CTA = True
            OFFLINE = False
        self.config = Config
        self.reader = SBTi.__new__(SBTi)
        self.reader.c = Config

    def frame(self, statuses):
        return pd.DataFrame({
            'company_name': ['Company ' + str(i) for i in range(len(statuses))],
            'isin': ['ISIN' + str(i) for i in range(len(statuses))],
            'lei': ['LEI00' + str(i) for i in range(len(statuses))],
            'near_term_status': statuses,
        })

    def test_current_and_previous_labels(self):
        data = self.frame(['Validated Targets', 'Targets set', 'Commitment to Set Targets',
                           'Committed', 'Commitment Removed', 'Commitment removed', None, ''])
        self.assertEqual(self.reader.filter_cta_file(data)['company_name'].tolist(),
                         ['Company 0', 'Company 1'])
        self.assertEqual(data.iloc[1]['near_term_status'], 'Targets set')

    def test_case_and_whitespace(self):
        data = self.frame(['  VALIDATED TARGETS  ', ' targets SET ', ' COMMITTED ', '   '])
        self.assertEqual(len(self.reader.filter_cta_file(data)), 2)

    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError, 'missing required columns: lei'):
            self.reader.filter_cta_file(self.frame(['Validated Targets']).drop(columns='lei'))

    def test_unknown_status(self):
        with self.assertRaisesRegex(ValueError, 'unknown near-term status values: New status'):
            self.reader.filter_cta_file(self.frame(['Validated Targets', 'New status']))

    def test_matching_lei_and_isin(self):
        self.reader.targets = self.frame(['Validated Targets', 'Targets set', 'Committed'])
        companies = [IDataProviderCompany(company_id=str(i), company_name='', isic='')
                     for i in range(4)]
        result = self.reader.get_sbti_targets(companies, {
            '0': ('unmatched', 'LEI000'), '1': ('ISIN1', None),
            '2': ('ISIN2', None), '3': ('absent', None),
        })
        self.assertEqual([c.sbti_validated for c in result], [True, True, False, False])

    def test_weighted_coverage_from_local_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cta.xlsx'
            self.frame(['Validated Targets', 'Targets set', 'Committed']).to_excel(path, index=False)
            portfolio = pd.DataFrame({
                'company_id': ['0', '1', '2', '3'],
                'company_isin': ['unmatched', 'ISIN1', 'ISIN2', 'absent'],
                'company_lei': ['LEI000', None, None, None],
                'investment_value': [10, 30, 20, 40],
            })
            coverage = PortfolioCoverageTVP(config=self.config, cta_file_path=str(path))
            self.assertAlmostEqual(coverage.get_portfolio_coverage(
                portfolio, PortfolioAggregationMethod.WATS), 40.0)

    def test_invalid_excel_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cta.xlsx'
            self.frame(['Unexpected']).to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                SBTi(config=self.config, cta_file_path=str(path))
