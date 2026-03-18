from typing import List, Type
import requests
import pandas as pd
import warnings
import os
import shutil


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

    def _get_bundled_cta_path(self):
        """
        Get the path to the bundled CTA file in the package
        """
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "inputs",
            "current-Companies-Taking-Action.xlsx",
        )

    def _use_bundled_fallback(self):
        """
        Copy the bundled CTA file to the target location as a fallback
        """
        bundled_path = self._get_bundled_cta_path()
        if os.path.isfile(bundled_path):
            try:
                target_dir = os.path.dirname(self.c.FILE_TARGETS)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(bundled_path, self.c.FILE_TARGETS)
                print(f'WARNING: Using bundled CTA file as fallback. Data may be outdated.')
                print(f'Copied bundled file to: {self.c.FILE_TARGETS}')
                return True
            except Exception as e:
                print(f'Error copying bundled CTA file: {e}')
                return False
        else:
            print(f'ERROR: Bundled CTA file not found at {bundled_path}')
            return False


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
            download_successful = False
            try:
                self._fetch_and_save_cta_file()
                download_successful = True
            
            except requests.HTTPError as err:
                if err.response.status_code == 403:
                    print(f'403 Error fetching the CTA file: {err}')
                else:
                    print(f'HTTP Error fetching the CTA file: {err}')
            except requests.RequestException as err:
                print(f'Network error fetching the CTA file: {err}')
                print(f'This may be due to firewall restrictions or network connectivity issues.')
            except Exception as err:
                print(f'Unexpected error fetching the CTA file: {err}')
            
            # If download failed and file still doesn't exist, use bundled fallback
            if not download_successful and not self._check_if_cta_file_exists():
                print(f'Attempting to use bundled CTA file as fallback...')
                if not self._use_bundled_fallback():
                    raise RuntimeError(
                        'Failed to download CTA file and no fallback available. '
                        'Please check network connectivity or set USE_LOCAL_CTA=True '
                        'with a valid FILE_TARGETS_CUSTOM_PATH.'
                    )
        else:            
            print(f'CTA file already exists in {self.c.FILE_TARGETS}, skipping download.')

    def _fetch_and_save_cta_file(self):
        # Ensure the directory exists
        target_dir = os.path.dirname(self.c.FILE_TARGETS)
        os.makedirs(target_dir, exist_ok=True)
        
        headers = {
            'x-request': 'download',
            'User-Agent': 'ITR-tool/0.9.2 (Python; ekonomi-finans@wwf.se)',
            'From': 'ekonomi-finans@wwf.se'
        }
        # read from the remote CTA file url
        response = requests.get(self.c.CTA_FILE_URL, headers=headers, timeout=30)
        # raise if the status code is not 200
        response.raise_for_status()

        with open(self.c.FILE_TARGETS, 'wb') as output:
            output.write(response.content)
            print(f'Successfully downloaded CTA file (Status: {response.status_code})')
            print(f'Saved to: {self.c.FILE_TARGETS}')

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
            # Handle both None and string values
            if lei is not None and str(lei).lower() not in ('nan', 'none', '') and len(str(lei)) > 3:
                targets = self.targets[
                    self.targets[self.c.COL_COMPANY_LEI] == lei
                ]
            elif isin is not None and str(isin).lower() not in ('nan', 'none', ''):
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

   