import ITR.data
from ITR import utils
from ITR.data.excel import ExcelProvider 
from ITR.data.sbti import SBTi 
from ITR.configs import PortfolioCoverageTVPConfig 

import os
import unittest
import pandas as pd


class TestSBTiData(unittest.TestCase):
    """
    Test various combinations of ISIN and LEI data.
    """

    def setUp(self) -> None:
        self.portfolios = [
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../",
                "inputs",
                "data_test_all_ISIN_LEI.csv"),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../",
                "inputs",
                "data_test_ISIN_no_LEI.csv"),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../",
                "inputs",
                "data_test_no_ISIN_all_LEI.csv"),
        ]
        self.provider = [ExcelProvider(path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../",
            "inputs",
            "data_test_local_CTA.xlsx",
            )
        )]

    def test_sbti_data_without_cta_files(self) -> None:
        """
        Test whether data is retrieved as expected from the local environment.
        """
        PortfolioCoverageTVPConfig.USE_LOCAL_CTA = True
        # Read a local mini version of the CTA file
        PortfolioCoverageTVPConfig.FILE_TARGETS_CUSTOM_PATH = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../"
            "inputs",
            "test_CTA.xlsx",
        )

        for portfolio in self.portfolios:
            # Read portfolio from csv file into dataframe
            portfolio = pd.read_csv(portfolio)
            # Convert dataframe to list of portfolio company objects
            portfolio = utils.dataframe_to_portfolio(portfolio)
            df_portfolio = pd.DataFrame.from_records(
                utils._flatten_user_fields(c) for c in portfolio)
            company_data = utils.get_company_data(self.provider, df_portfolio["company_id"].tolist())

            # Get SBTi data
            company_data = SBTi().get_sbti_targets(company_data, utils._make_id_map(df_portfolio))

            # Check that the data is as expected
            self.assertEqual(len(company_data), 3)

    def test_sbti_data_skipping_file(self) -> None:
        """
        Test whether data is retrieved as expected from the SBTi wbesite but 
        skipping in case of existent file.
        """
        PortfolioCoverageTVPConfig.SKIP_CTA_FILE_IF_EXISTS = True

        for portfolio in self.portfolios:
            # Read portfolio from csv file into dataframe
            portfolio = pd.read_csv(portfolio)
            # Convert dataframe to list of portfolio company objects
            portfolio = utils.dataframe_to_portfolio(portfolio)
            df_portfolio = pd.DataFrame.from_records(
                utils._flatten_user_fields(c) for c in portfolio)
            company_data = utils.get_company_data(self.provider, df_portfolio["company_id"].tolist())

            # Get SBTi data
            company_data = SBTi().get_sbti_targets(company_data, utils._make_id_map(df_portfolio))

            # Check that the data is as expected
            self.assertEqual(len(company_data), 3)

    def test_sbti_data(self) -> None:
        """
        Test whether data is retrieved as expected from the SBTi wbesite.
        Also test that ISIN and LEI data are treated correctly in _make_id_map.
        """
        for portfolio in self.portfolios:
            # Read portfolio from csv file into dataframe
            portfolio = pd.read_csv(portfolio)
            # Convert dataframe to list of portfolio company objects
            portfolio = utils.dataframe_to_portfolio(portfolio)
            df_portfolio = pd.DataFrame.from_records(
                utils._flatten_user_fields(c) for c in portfolio)
            company_data = utils.get_company_data(self.provider, df_portfolio["company_id"].tolist())
            target_data = utils.get_targets(self.provider, df_portfolio["company_id"].tolist())
            
            # Get SBTi data
            company_data = SBTi().get_sbti_targets(company_data, utils._make_id_map(df_portfolio))
            

            # Check that the data is as expected
            self.assertEqual(len(company_data), 3)


if __name__ == "__main__":
    test = TestSBTiData()
    test.setUp()
    test.test_sbti_data()