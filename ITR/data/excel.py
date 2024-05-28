from typing import Type, List
from pydantic import ValidationError
from datetime import date
import logging

import pandas as pd
from ITR.data.data_provider import DataProvider
from ITR.configs import ColumnsConfig
from ITR.interfaces import IDataProviderCompany, IDataProviderTarget, S3Category


class ExcelProvider(DataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param config: A dictionary containing a "path" field that leads to the path of the CSV file
    """

    def __init__(self, path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.data = pd.read_excel(path, sheet_name=None, skiprows=0)
        # Set all missing coverage values to 0.0
        self.data['target_data'][['coverage_s1', 'coverage_s2', 'coverage_s3', 'reduction_ambition']] = \
            self.data['target_data'][['coverage_s1', 'coverage_s2', 'coverage_s3', 'reduction_ambition']].fillna(0.0)
       
        try:
            self.data['target_data']['s3_category'] = self.data['target_data']['s3_category'].apply(lambda x: int(x) if str(x).isdigit() and 1 <= int(x) <= 15 else 0)
        except Exception as e:
            print(f"An error occurred: {e}")
       
        self.c = config

    def get_targets(self, company_ids: List[str]) -> List[IDataProviderTarget]:
        """
        Get all relevant targets for a list of company ids (ISIN). This method should return a list of
        IDataProviderTarget instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the targets
        """
        model_targets = self._target_df_to_model(self.data["target_data"])
        model_targets = [
            target for target in model_targets if target.company_id in company_ids
        ]
        return model_targets

    def _target_df_to_model(self, df_targets):
        """
        transforms target Dataframe into list of IDataProviderTarget instances

        :param df_targets: pandas Dataframe with targets
        :return: A list containing the targets
        """
        logger = logging.getLogger(__name__)
         # 1) Check if 'statement_date' looks like a date
        df_targets['statement_date'] = pd.to_datetime(df_targets['statement_date'], format='%Y', errors='coerce')

        # 2) If 'statement_date' is empty, check 'start_year'
        df_targets.loc[df_targets['statement_date'].isna(), 'statement_date'] = pd.to_datetime(df_targets['start_year'], format='%Y', errors='coerce')

        # 3) If 'start_year' is empty, use 'base_year'
        df_targets.loc[df_targets['statement_date'].isna(), 'statement_date'] = pd.to_datetime(df_targets['base_year'], format='%Y', errors='coerce')
        targets = df_targets.to_dict(orient="records")
        model_targets: List[IDataProviderTarget] = []
        for target in targets:
            try:
                # Map the s3_category values to the S3Category enum
                target['s3_category'] = S3Category(target['s3_category']) 
                model_targets.append(IDataProviderTarget.parse_obj(target))
            except ValidationError as e:
                print(f"Validationerror: {e}")
                logger.warning(
                    f"(one of) the target(s) {target[self.c.TARGET_IDS]} of company {target[self.c.COMPANY_NAME]} is invalid and will be skipped"
                    
                )
                pass
        return model_targets

    def get_company_data(self, company_ids: List[str]) -> List[IDataProviderCompany]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of IDataProviderCompany
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        data_company = self.data["fundamental_data"]
        # Pydantic doesn't accept NaN values in string fields, 
        # so we need to convert them to empty strings
        fields_to_convert = ['isic','country', 'sector', 'industry_level_1', 'industry_level_2', 
                             'industry_level_3', 'industry_level_4']
        data_company[fields_to_convert] = data_company[fields_to_convert].fillna('')

        companies = data_company.to_dict(orient="records")
        model_companies: List[IDataProviderCompany] = [
            IDataProviderCompany.parse_obj(company) for company in companies
        ]
        for company in model_companies:
            if company.ghg_s1 is not None and company.ghg_s2 is not None:
                company.ghg_s1s2 = company.ghg_s1 + company.ghg_s2

        model_companies = [
            target for target in model_companies if target.company_id in company_ids
        ]
        return model_companies

    def get_sbti_targets(self, companies: list) -> list:
        """
        For each of the companies, get the status of their target (Target set, Committed or No target) as it's known to
        the SBTi.

        :param companies: A list of companies. Each company should be a dict with a "company_name" and "company_id"
                            field.
        :return: The original list, enriched with a field called "sbti_target_status"
        """
        raise NotImplementedError
