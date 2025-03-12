from typing import List, Type
import requests
import pandas as pd
import warnings
import os


from ITR.configs import PortfolioCoverageTVPConfig
from ITR.interfaces import IDataProviderCompany


class SBTi:
    """
    Data provider skeleton for SBTi. This class only provides the sbti_validated field for existing companies.
    """

    def _check_if_cta_file_exists(self):
        """
        Check if the CTA file exists in the local file system
        """
        return os.path.isfile(self.c.FILE_TARGETS)


    def _check_CTA_less_than_one_week_old(self):
        """
        Check if the CTA file is older than a week
        """
        file_updated_time = os.path.getmtime(self.c.FILE_TARGETS)
        week_in_seconds = 7 * 24 * 60 * 60
        # If the file was updated more than a week ago, return False
        if file_updated_time < pd.Timestamp.now().timestamp() - week_in_seconds:
            return False
        else:
            return True

    def _use_local_cta_file(self):
        if self.c.FILE_TARGETS_CUSTOM_PATH is None:
            raise ValueError('Please set FILE_TARGETS_CUSTOM_PATH to the path of the CTA file.')
        self.c.FILE_TARGETS = self.c.FILE_TARGETS_CUSTOM_PATH

        # check that file is not more than a week old
        if  self._check_if_cta_file_exists():
            # file_age = os.path.getmtime(self.c.FILE_TARGETS)
            # week_in_seconds = 7 * 24 * 60 * 60 # frequency of CTA file updates

            if not self._check_CTA_less_than_one_week_old(): 
                print(f'CTA file is older than a week, if you want to keep your file up-to-date please update the file at {self.c.FILE_TARGETS}.')
        else:
            raise ValueError('CTA file does not exist')

    def _download_cta_file(self):
        get_file = False
        if self._check_if_cta_file_exists():
            if self.c.SKIP_CTA_FILE_IF_EXISTS:
                if not self._check_CTA_less_than_one_week_old():
                    get_file = True
            else:
                get_file = True
        else:                  
            get_file = True

        if get_file:
            try:
                self._fetch_and_save_cta_file()
            
            except requests.HTTPError as err:
                if err.response.status_code == 403:
                    print(f'403 Error fetching the CTA file: {err}')
                else:
                    print(f'Error fetching the CTA file: {err}')
        else:            
            print(f'CTA file already exists in {self.c.FILE_TARGETS}, skipping download.')

    def _fetch_and_save_cta_file(self):
        try:
            headers = {
                'x-request': 'download',
                'User-Agent': 'ITR-tool/0.9.2 (Python; ekonomi-finans@wwf.se)',
                'From': 'ekonomi-finans@wwf.se'
            }
            # read from the remote CTA file url
            response = requests.get(self.c.CTA_FILE_URL, headers=headers)
            # raise if the status code is not 200
            response.raise_for_status()

            with open(self.c.FILE_TARGETS, 'wb') as output:
                output.write(response.content)
                print(f'Status code from fetching the CTA file: {response.status_code}, 200 = OK')
        except requests.HTTPError as err:
            print(f'Error fetching the CTA file: {err}')

    def handle_cta_file(self):
        if self.c.USE_LOCAL_CTA:
            self._use_local_cta_file()
        else:
            self._download_cta_file()

    def __init__(
        self, config: Type[PortfolioCoverageTVPConfig] = PortfolioCoverageTVPConfig
    ):
        self.c = config

        # Handle CTA file
        self.handle_cta_file()

        # Read CTA file into pandas dataframe
        # Suppress warning about openpyxl - check if this is still needed in the released version.
        warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
       
        path = os.path.realpath(self.c.FILE_TARGETS)
        self.targets = pd.read_excel(path)

        
    def filter_cta_file(self, targets):
        """
        Filter the CTA file to create a dataframe that has one row per company
        with the columns "Action" and "Target".
        If Action = Target then only keep the rows where Target = Near-term.
        """

        # Create a new dataframe with only the columns "Action" and "Target"
        # and the columns that are needed for identifying the company
        targets = targets[
            [
                self.c.COL_COMPANY_NAME, 
                self.c.COL_COMPANY_ISIN, 
                self.c.COL_COMPANY_LEI, 
                self.c.COL_TARGET
            ]
        ]
        
        # Keep rows where Action = Target and Target = Near-term 
        df_nt_targets = targets[
           (targets[self.c.COL_TARGET] == self.c.VALUE_STATUS_SET)]
        
        # Drop duplicates in the dataframe by waterfall. 
        # Do company name last due to risk of misspelled names
        # First drop duplicates on LEI, then on ISIN, then on company name
        df_nt_targets = pd.concat([
            df_nt_targets[~df_nt_targets[self.c.COL_COMPANY_LEI].isnull()].drop_duplicates(
                subset=self.c.COL_COMPANY_LEI, keep='first'
            ), 
            df_nt_targets[df_nt_targets[self.c.COL_COMPANY_LEI].isnull()]
        ])
        
        df_nt_targets = pd.concat([
            df_nt_targets[~df_nt_targets[self.c.COL_COMPANY_ISIN].isnull()].drop_duplicates(
                subset=self.c.COL_COMPANY_ISIN, keep='first'
            ),
            df_nt_targets[df_nt_targets[self.c.COL_COMPANY_ISIN].isnull()]
        ])

        df_nt_targets.drop_duplicates(subset=self.c.COL_COMPANY_NAME, inplace=True)
  
        return df_nt_targets
    
    def get_sbti_targets(
        self, companies: List[IDataProviderCompany], id_map: dict
    ) -> List[IDataProviderCompany]:
        """
        Check for each company if they have an SBTi validated target, first using the company LEI, 
        if available, and then using the ISIN.
        
        :param companies: A list of IDataProviderCompany instances
        :param id_map: A map from company id to a tuple of (ISIN, LEI)
        :return: A list of IDataProviderCompany instances, supplemented with the SBTi information
        """
        # Filter out information about targets
        self.targets = self.filter_cta_file(self.targets)

        for company in companies:
            id_tuple = id_map.get(company.company_id)
            if id_tuple is None:
                continue
            isin, lei = id_tuple
            # Check lei and length of lei to avoid zeros 
            if not lei.lower() == 'nan' and len(lei) > 3:
                targets = self.targets[
                    self.targets[self.c.COL_COMPANY_LEI] == lei
                ]
            elif not isin.lower() == 'nan':
                targets = self.targets[
                    self.targets[self.c.COL_COMPANY_ISIN] == isin
                ]
            else:
                continue
            if len(targets) > 0:
                company.sbti_validated = (
                    self.c.VALUE_STATUS_SET in targets[self.c.COL_TARGET].values
                )
        return companies

   