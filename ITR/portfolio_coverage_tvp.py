from typing import Type, Optional
import pandas as pd
from ITR.configs import PortfolioCoverageTVPConfig, ColumnsConfig
from ITR.interfaces import IDataProviderCompany
from ITR.portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod
from ITR.data.sbti import SBTi


class PortfolioCoverageTVP(PortfolioAggregation):
    """
    Lookup the companies in the given portfolio and determine whether they have a SBTi approved target.

    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the PortfolioCoverageTVPConfig
                    class and overwriting one of the parameters.
    :param cta_file_path: Optional path to a user-provided CTA file. If set, the file will be used instead of
                    downloading from the internet.
    """

    def __init__(
        self,
        config: Type[PortfolioCoverageTVPConfig] = PortfolioCoverageTVPConfig,
        cta_file_path: Optional[str] = None,
    ):
        super().__init__(config)
        self.c: Type[PortfolioCoverageTVPConfig] = config
        self.cta_file_path = cta_file_path

    def _enrich_sbti_validated(self, company_data: pd.DataFrame) -> pd.DataFrame:
        """
        Load the CTA file and enrich the company data with the sbti_validated status.
        Skipped if sbti_validated is already populated in the data.
        """
        # If sbti_validated is already present and has any True values, assume it's pre-populated
        if (
            self.c.COLS.SBTI_VALIDATED in company_data.columns
            and company_data[self.c.COLS.SBTI_VALIDATED].any()
        ):
            return company_data

        # Ensure ISIN and LEI columns exist for SBTi matching
        for col in [ColumnsConfig.COMPANY_ISIN, ColumnsConfig.COMPANY_LEI]:
            if col not in company_data.columns:
                company_data[col] = None

        # Build ISIN/LEI map from the portfolio data
        id_map = {}
        unique_companies = company_data[
            [ColumnsConfig.COMPANY_ID, ColumnsConfig.COMPANY_ISIN, ColumnsConfig.COMPANY_LEI]
        ].drop_duplicates(subset=ColumnsConfig.COMPANY_ID)
        for _, row in unique_companies.iterrows():
            id_map[row[ColumnsConfig.COMPANY_ID]] = (
                row.get(ColumnsConfig.COMPANY_ISIN),
                row.get(ColumnsConfig.COMPANY_LEI),
            )

        # Load CTA data and enrich sbti_validated
        sbti = SBTi(config=self.c, cta_file_path=self.cta_file_path)
        # Build lightweight company objects with only the fields SBTi matching needs
        companies_for_sbti = [
            IDataProviderCompany(company_id=cid, company_name="", isic="")
            for cid in company_data[ColumnsConfig.COMPANY_ID].unique()
        ]
        companies_for_sbti = sbti.get_sbti_targets(companies_for_sbti, id_map)
        sbti_map = {c.company_id: c.sbti_validated for c in companies_for_sbti}
        company_data[self.c.COLS.SBTI_VALIDATED] = (
            company_data[self.c.COLS.COMPANY_ID].map(sbti_map).fillna(False)
        )

        return company_data

    def get_portfolio_coverage(
        self,
        company_data: pd.DataFrame,
        portfolio_aggregation_method: PortfolioAggregationMethod,
    ) -> Optional[float]:
        """
        Get the TVP portfolio coverage (i.e. what part of the portfolio has a SBTi validated target).
        Loads the CTA file and enriches company data with sbti_validated status before computing coverage,
        unless sbti_validated is already populated.

        :param company_data: The company as it is returned from the data provider's get_company_data call.
        :param portfolio_aggregation_method: PortfolioAggregationMethod: The aggregation method to use
        :return: The aggregated score
        """
        company_data = self._enrich_sbti_validated(company_data)

        company_data[self.c.OUTPUT_TARGET_STATUS] = company_data.apply(
            lambda row: 100 if row[self.c.COLS.SBTI_VALIDATED] else 0, axis=1
        )

        return self._calculate_aggregate_score(
            company_data, self.c.OUTPUT_TARGET_STATUS, portfolio_aggregation_method
        ).sum()
