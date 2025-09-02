# This is target_validation.py
import datetime
import itertools
import logging
import copy

import pandas as pd
from typing import Type, List, Tuple, Optional, cast
from ITR.configs import PortfolioAggregationConfig, TemperatureScoreConfig

from ITR.interfaces import (
    IDataProviderTarget,
    IDataProviderCompany,
    EScope,
    ETimeFrames,
    S3Category
)

logger = logging.getLogger(__name__)


class TargetProtocol:
    """
    This class validates the targets, to make sure that only active, useful 
    targets are considered. It then combines the targets with company-related 
    data into a dataframe where there's one row for each of the nine possible 
    target types (short, mid, long * S1+S2, S3, S1+S2+S3). 
    This class follows the procedures outlined by the target protocol that is 
    a part of the "Temperature Rating Methodology" (2024), which has been created 
    by CDP Worldwide and WWF International.

    :param config: A Portfolio aggregation config
    """

    def __init__(
        self, config: Type[PortfolioAggregationConfig] = PortfolioAggregationConfig
    ):
        self.c = config
        self.logger = logging.getLogger(__name__)
        self.target_data: pd.DataFrame = pd.DataFrame()
        self.intensity_metric_types = TemperatureScoreConfig.intensity_metric_types
        self.company_data: pd.DataFrame = pd.DataFrame()
        self.data: pd.DataFrame = pd.DataFrame()

    def process(
        self, targets: List[IDataProviderTarget], companies: List[IDataProviderCompany]
    ) -> pd.DataFrame:
        """
        Process the targets and companies, validate all targets and return a data frame that combines all targets and company data into a 9-box grid.

        :param targets: A list of targets
        :param companies: A list of companies
        :return: A data frame that combines the processed data
        """
        logger.info(f"started processing {len(targets)=} and {len(companies)=}")
        # Create multiindex on company, timeframe and scope for performance later on
        self.company_data = pd.DataFrame.from_records([c.dict() for c in companies])
        targets = self._prepare_targets(targets)
        self.target_data = pd.DataFrame.from_records([c.dict() for c in targets])
        self.target_data['statement_date'] = pd.to_datetime(self.target_data['statement_date'])

        # Create an indexed DF for performance purposes
        self.target_data.index = (
            self.target_data.reset_index()
            .set_index(
                [self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]
            )
            .index
        )
        self.target_data = self.target_data.sort_index()

        self.target_data = self.sort_on_vintage(self.target_data)

        self.target_data = self.sort_boundary_coverage(self.target_data)

        self.target_data['reduction_ambition'] = self.target_data.apply(self._scale_reduction_ambition_by_boundary_coverage_new, axis=1)

        self.group_targets()
        
        #Drop the column sbti_validated from target_data since it's no longer needed
        self.data = self.data.drop(columns=["sbti_validated"], errors="ignore")
        out =  pd.merge(
            left=self.data, right=self.company_data, how="outer", on=["company_id"]
        )
        logger.info(f"finished processing {out.shape=}")
        return out

    def _validate(self, target: IDataProviderTarget) -> bool:
        """
        Validate a target, meaning it should:

        * Have a valid type
        * Not be finished
        * A valid end year

        :param target: The target to validate
        :return: True if it's a valid target, false if it isn't
        """
        # If the target is set using the CDP_WWF temperature scoring methodology
        if target.target_type.lower() == "t_score":
            return self._validate_t_score(target)
        # Only absolute targets or intensity targets with a valid intensity metric are allowed.
        # As of v1.5 we have target type 'int_to_abs' which is an intensity target that has been converted to an absolute target
        target_type = (
            "abs" in target.target_type.lower()
            or "int_to_abs" in target.target_type.lower()
            or (
                "intensity" in target.target_type.lower()
                and target.intensity_metric in self.intensity_metric_types
            )
        )
        # There must be a number in the reduction ambition field, even zero is okay.
        target_reduction_ambition = not pd.isna(target.reduction_ambition)

        # The target should not have achieved its reduction yet.
        target_progress = (
            pd.isnull(target.achieved_reduction)
            or target.achieved_reduction is None
            or target.achieved_reduction < 1
        )

        # The end year should be greater than the start year.
        if target.start_year is None or pd.isnull(target.start_year):
            target.start_year = target.base_year

        target_end_year = target.end_year > target.start_year

        # The end year should be greater than or equal to the current year
        target_current = target.end_year >= datetime.datetime.now().year

        # Check that base year ghg data is available for the scope of the target
        s1 = target.scope != EScope.S1 or (
            pd.notna(target.coverage_s1)
            and pd.notna(target.base_year_ghg_s1) 
        )
        s2 = target.scope != EScope.S2 or (
            pd.notna(target.coverage_s2)
            and pd.notna(target.base_year_ghg_s2)  
        )
        s1s2 = target.scope != EScope.S1S2 or (
            pd.notna(target.base_year_ghg_s1)
            and pd.notna(target.base_year_ghg_s2) 
        )
        # Note that base year s3 ghg is checked in the method _find_s3_targets
        # for individual s3 category targets that have been split from s1s2s3 targets
        s1s2s3 = target.scope != EScope.S1S2S3 or (
            pd.notna(target.base_year_ghg_s1)
            and pd.notna(target.base_year_ghg_s2) 
        )
        s3 = target.scope != EScope.S3 or (
            pd.notna(target.coverage_s3)
            and pd.notna(target.base_year_ghg_s3)
        )
        return (
            target_type
            and target_reduction_ambition 
            and target_progress 
            and target_end_year 
            and target_current 
            and s1 
            and s2
            and s1s2
            and s1s2s3
            and s3
        )
    
    def _validate_t_score(self, target: IDataProviderTarget) -> bool:
        """
        Validate targets set using the CDP_WWF temperature scoing methodology
        :param target: The target to validate
        :return: True if it's a valid target, false if it isn't
        """
        # The end year should be greater than the start year.
        if target.start_year is None or pd.isnull(target.start_year):
            target.start_year = target.base_year

        target_end_year = target.end_year > target.start_year

        # The end year should be greater than or equal to the current year
        # Added in update Oct 22
        target_current = target.end_year >= datetime.datetime.now().year

        # Target scope must be S3
        target_scope = target.scope == EScope.S3
        
        return (
            target_end_year
            and target_current
            and target_scope
        )

    def _split_s1s2s3(self,
        target: IDataProviderTarget,
    ) -> Tuple[IDataProviderTarget, Optional[IDataProviderTarget]]:
        """
        If there is a s1s2s3 scope, split it into two targets with s1s2 and s3
        This S3 target becomes a headline target
        :param target: The input target
        :return The split targets or the original target and None
        """
        if target.target_type.lower() == "t_score":
            return target, None
        if target.scope == EScope.S1S2S3:
            s1s2, s3 = target.copy(), None
            s1s2.s3_category = S3Category.N_A
            # Assign scalar values to the new DataFrames
            s1s2.base_year_ghg_s1 = cast(float, target.base_year_ghg_s1)
            s1s2.base_year_ghg_s2 = cast(float, target.base_year_ghg_s2)
            if (
                not pd.isnull(target.base_year_ghg_s1)
                and not pd.isnull(target.base_year_ghg_s2)
            ) or target.coverage_s1 == target.coverage_s2:
                s1s2.scope = EScope.S1S2
                if (
                    pd.notna(s1s2.coverage_s1)
                    and pd.notna(s1s2.coverage_s2)
                    and pd.notna(s1s2.base_year_ghg_s1)
                    and pd.notna(s1s2.base_year_ghg_s2)
                    and s1s2.base_year_ghg_s1 + s1s2.base_year_ghg_s2 != 0
                ):
                    coverage_percentage = (
                        s1s2.coverage_s1 * s1s2.base_year_ghg_s1
                        + s1s2.coverage_s2 * s1s2.base_year_ghg_s2
                    ) / (s1s2.base_year_ghg_s1 + s1s2.base_year_ghg_s2)
                    s1s2.coverage_s1 = coverage_percentage
                    s1s2.coverage_s2 = coverage_percentage

            if not pd.isnull(target.coverage_s3):
                s3 = target.copy()
                s3.scope = EScope.S3
                if s3.s3_category == S3Category.N_A:
                    s3.s3_category = S3Category.CAT_H_LINE
            return s1s2, s3
                
        else:
            return target, None
           
    def _split_s1s2_new(self,
        target: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Split the target into two targets, one for the S1 data and one for the S2 data.

        :param target: The target to potentially split.
        :return: A list containing the S1 target and the S2 target from the split.
        """
        if target[self.c.COLS.TARGET_REFERENCE_NUMBER].iloc[0].lower() == "t_score":
            return target
        targets = target.iloc[0].copy()
        # before splitting S1S2 targets we need to verify that there is GHG data to aggregate the scores later
        # TODO - verify that company is one unique row - for future cleaning module
        company = self.company_data[self.company_data[self.c.COLS.COMPANY_ID] == targets.company_id]
        if (not (pd.isna(company[self.c.COLS.GHG_SCOPE1].values[0])
            or pd.isna(company[self.c.COLS.GHG_SCOPE2].values[0]))
            and target.scope.iloc[0] == EScope.S1S2
        ):
            s1 = targets.copy()
            s2 = targets.copy()           

            # Assign scalar values to the new DataFrames
            s1['coverage_s1'] = targets['coverage_s1']
            s2['coverage_s2'] = targets['coverage_s2']
            s1['reduction_ambition'] = targets['reduction_ambition']
            s2['reduction_ambition'] = targets['reduction_ambition']
        
            s1['scope'] = EScope.S1
            s2['scope'] = EScope.S2
            # Append '_1' and '_2' to the target_ids of s1 and s2, respectively
            s1['target_ids'] = [id_ + '_1' for id_ in s1['target_ids']]
            s2['target_ids'] = [id_ + '_2' for id_ in s2['target_ids']]
      
            return pd.DataFrame([s1, s2])
        return target
        
    @staticmethod
    def _scale_reduction_ambition_by_boundary_coverage_new(
        target: pd.DataFrame,
        ) -> float:
        """
        Change in ITR method 1.5: all targets have their ambition scaled by their boundary coverage.
        :param target: The input target
        :return: The original target with a weighted reduction ambition, if so required
        """
                
        if pd.isna(target.reduction_ambition):
            reduction_ambition = 0.0
        
        elif target.scope == EScope.S1:
            reduction_ambition = (
                target.reduction_ambition * target.coverage_s1
            )
        elif target.scope == EScope.S2:
            reduction_ambition = (
                target.reduction_ambition * target.coverage_s2
            )
        elif target.scope == EScope.S3:
            reduction_ambition = (
                target.reduction_ambition * target.coverage_s3
            )
        elif target.scope == EScope.S1S2:
            if pd.isna(target.coverage_s1):
                target.coverage_s1 = 0.0
            if pd.isna(target.coverage_s2):
                target.coverage_s2 = 0.0
            if (
                not pd.isnull(target.base_year_ghg_s1)
                and not pd.isnull(target.base_year_ghg_s2)
                and target.base_year_ghg_s1 + target.base_year_ghg_s2 != 0):
                    combined_coverage = (
                        target.coverage_s1 * target.base_year_ghg_s1
                        + target.coverage_s2 * target.base_year_ghg_s2
                    ) / (target.base_year_ghg_s1 + target.base_year_ghg_s2)
  
                    target.coverage_s1 = combined_coverage
                    target.coverage_s2 = combined_coverage
                    reduction_ambition = (
                        target.reduction_ambition * combined_coverage
                    )
            else:
                reduction_ambition = 0.0
        else:
            reduction_ambition = target.reduction_ambition

        return reduction_ambition # type: ignore
    @staticmethod
    def _assign_time_frame(target: IDataProviderTarget) -> IDataProviderTarget:
        """
        Time frame is forward looking: target year - current year. 
            Less than 5y = short, 
            between 5 and 10 is mid, 
            more than 10 is long

        :param target: The input target
        :return: The original target with the time_frame field filled out (if so required)
        """
        now = datetime.datetime.now()
        time_frame = target.end_year - now.year
        # Method 6.3.6.1: if target is of type T_SCORE and 
        # the company is validated by sbti, then add five years to timeframe
        if (target.target_type.lower() == "t_score") and target.sbti_validated:
            time_frame += 5

        if time_frame < 5:
            target.time_frame = ETimeFrames.SHORT
        elif time_frame <= 10:
            target.time_frame = ETimeFrames.MID
        elif time_frame > 10:
            target.time_frame = ETimeFrames.LONG

        return target

    def _prepare_targets(self, targets: List[IDataProviderTarget]):
        """
        logic
            - drop invalid targets
            - identifying the pure-S2 targets for later use
            - splitting s1s2s3 into s1s2 and s3
            - combining s1 and s2
            - assign target.reduction_ambition by considering target's boundary coverage
 
        :param targets:
        :return:
        """
        target_input_count = len(targets)
        targets = list(filter(self._validate, targets))
        logger.info(f"dropped {(target_input_count - len(targets))=:,} invalid targets")
        targets = [self._assign_time_frame(target) for target in targets]

        targets = list(
            filter(
                None, itertools.chain.from_iterable(map(self._split_s1s2s3, targets))
            )
        )
        
        return targets

    def _find_target(self, row: pd.Series, target_columns: List[str]) -> pd.DataFrame:
       
        """
        Find the target that corresponds to a given row. If there are multiple targets available, filter them.

        :param row: The row from the data set that should be looked for
        :param target_columns: The columns to return
        :return: records from the input data, which contains company and target information, that meet specific criteria. For example, record of greatest emissions_in_scope
        """
        try:

            target_data = self.target_data.xs(
                (
                    row[self.c.COLS.COMPANY_ID],
                    row[self.c.COLS.TIME_FRAME],
                    row[self.c.COLS.SCOPE],
                )
            ).copy() # type: ignore
            if isinstance(target_data, pd.Series):
                # One match with Target data
                result_df = pd.DataFrame([target_data], columns=target_columns)

            else:
                if target_data.scope.iloc[0] == EScope.S3:
                    coverage_column = self.c.COLS.COVERAGE_S3
                elif target_data.scope.iloc[0] == EScope.S2:
                    coverage_column = self.c.COLS.COVERAGE_S2
                else:
                    coverage_column = self.c.COLS.COVERAGE_S1
                # In case more than one target is available; we prefer targets with 
                # later confirmation date, 
                # higher coverage,
                # later end year, and target type 'absolute'
                
                # We also prefer longer time spans within the same time frame
                target_data['END_YEAR_MINUS_BASE_YEAR'] = (
                    target_data[self.c.COLS.END_YEAR] 
                    - target_data[self.c.COLS.BASE_YEAR]
                )
                # Reduction ambition is measured by CAR
                try:
                    target_data['CAR'] = target_data.apply(
                        lambda row: (
                            1.0 if row[self.c.COLS.REDUCTION_AMBITION] >= 1 else abs(
                                (1 - row[self.c.COLS.REDUCTION_AMBITION]) ** 
                                (1 / row['END_YEAR_MINUS_BASE_YEAR']) - 1
                            )
                        ),
                        axis=1
                    )
                except ZeroDivisionError:
                    target_data['CAR'] = 0.0
                # Scope 3 targets need to be filtered separately
                if target_data.scope.iloc[0] != EScope.S3:
                    target_data = (
                        target_data.sort_values(
                            by=[
                                self.c.COLS.TARGET_CONFIRM_DATE,
                                coverage_column,
                                self.c.COLS.TARGET_REFERENCE_NUMBER,
                                'CAR',
                                'END_YEAR_MINUS_BASE_YEAR',
                                self.c.COLS.END_YEAR,
                            ],
                            axis=0,
                            ascending=[False, False, True, False,False, False ],
                        ).iloc[0][target_columns]
                    )
                    result_df = pd.DataFrame([target_data], columns=target_columns)

                else:

                    result_df = self._find_s3_targets(target_data, target_columns)
                    if result_df.empty:
                        result_df = pd.DataFrame([row], columns=target_columns)
                        result_df['to_calculate'] = False 
                        return result_df

            result_df.loc[:, 'to_calculate'] = True
            return result_df
                          
        except KeyError:
            # No target found
            result_df = pd.DataFrame([row], columns=target_columns)
            result_df['to_calculate'] = False # TS for selected targets are not to be calculated
            return result_df
    
    def _find_s3_targets(self, target_data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """
        Find S3 target that correspond to the given row. Note that there may be more
        than one S3 target. We first look for a headline target and return that. 
        If there is none we look for the targets with the latest confirmation date.
        Then check if there is more than one target with the same confirmation date.
        The method then returns all non headline targets with the latest confirmation date.

        :param target_data: The target data
        :param target_columns: The columns to return
        :return: The target data that meet the criteria
        """
        selected_targets = []
        headline_target = target_data[target_data['s3_category'] == S3Category.CAT_H_LINE]
        if not headline_target.empty:
            # Sort by vintage and cope 3 coverage 
            headline_target = headline_target.sort_values(
                                        by=[
                                            self.c.COLS.TARGET_CONFIRM_DATE, 
                                            self.c.COLS.COVERAGE_S3,
                                            self.c.COLS.TARGET_REFERENCE_NUMBER,
                                            'CAR',
                                            'END_YEAR_MINUS_BASE_YEAR',
                                            ], 
                                            axis=0,
                                            ascending=[False, False, True, False, False])           
        
            selected_targets.append(headline_target.head(1)) 
        else: # If there is no headline target, we loop through the categories and select the best target
            filtered_targets = target_data[target_data['s3_category'].isin(
                [cat for cat in S3Category if cat not in [S3Category.CAT_H_LINE, S3Category.N_A]]
                )]
            # Invalidate targets with no base year GHG data
            filtered_targets = filtered_targets.dropna(subset=['base_year_ghg_s3'])
            if filtered_targets.empty:
                return pd.DataFrame(columns=target_columns)
            sorted_targets = filtered_targets.sort_values(
                                    by=[
                                        self.c.COLS.TARGET_CONFIRM_DATE, 
                                        self.c.COLS.COVERAGE_S3, 
                                        self.c.COLS.TARGET_REFERENCE_NUMBER,
                                        'CAR',
                                        'END_YEAR_MINUS_BASE_YEAR',
                                        ],
                                        ascending=[False, False, True, False, False])
           
        # Select the top target for each category
            top_targets = sorted_targets.groupby('s3_category').head(1)
            selected_targets.append(top_targets)

        if selected_targets:
            selected_targets_df = pd.concat(selected_targets, ignore_index=True)
            final = selected_targets_df[target_columns]
        else:
            final = pd.DataFrame(columns=target_columns)
    
        return final
    
    def group_targets(self):
        """
        Group the targets and create the target grid (short, mid, long * s1s2, s3, s1s2s3).
        Group valid targets by category & filter multiple targets
        Input: a list of valid targets for each company:
        For each company:

        Group all valid targets based on scope (S1+S2 / S3 / S1+S2+S3) and time frame (short / mid / long-term)
        into 6 categories.

        For each category: if more than 1 target is available, filter based on the following criteria
        -- Latest vintage
        -- Highest boundary coverage
        -- Latest end year
        -- Target type: Absolute over intensity
        -- If all else is equal: average the ambition of targets
        """
        self.target_data = self.target_data.sort_index(level=self.target_data.index.names)

        grid_columns = [
            self.c.COLS.COMPANY_ID,
            self.c.COLS.TIME_FRAME,
            self.c.COLS.SCOPE,
        ]
        companies = self.company_data[self.c.COLS.COMPANY_ID].unique()
        #scopes = [EScope.S1S2, EScope.S3, EScope.S1S2S3]
        scopes = [EScope.S1, EScope.S2, EScope.S1S2, EScope.S3, EScope.S1S2S3]
        empty_columns = [
            column for column in self.target_data.columns if column not in grid_columns
        ]
        extended_data = pd.DataFrame(
            list(
                itertools.product(
                    *[companies, ETimeFrames, scopes] + [[None]] * len(empty_columns)
                )
            ),
            columns=grid_columns + empty_columns,
        )

        target_columns = extended_data.columns.tolist() 
        results = []
        for _, row in extended_data.iterrows():
            result = self._find_target(row, target_columns)
            results.append(result)
        # Remove empty or all-NA columns from each DataFrame in results (pandas futureproofing)
        results = [df.dropna(how='all', axis=1) for df in results]
        self.data = pd.concat(results, ignore_index=True)

        return self.data
    
    def sort_on_vintage(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """
        For each combination of company id, time frame, and scope, find all targets with the same latest 
        TARGET_CONFIRM_DATE and return as a DataFrame of targets.

        :param target_data: a DataFrame with target data
        :return: A DataFrame of targets sorted based on their vintage
        """
        target_data['statement_year'] = target_data['statement_date'].dt.year

        # Find the latest year for each group
        latest_years = target_data.groupby(
            level=[self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]
        )['statement_year'].transform('max')

        latest_data = target_data[target_data['statement_year'] == latest_years]

        latest_data = latest_data.drop(columns=['statement_year'])

        return latest_data
    
    def sort_boundary_coverage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        We want to select the scope 1 and scope 2 targets that have the highest
        combined boundary coverage for each group of self.c.COLS.COMPANY_ID, 
        self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE. 
        We compare combinations of individual scope 1 and scope 2
        targets with combined S1+S2 targets, and select the combination with 
        the highest boundary coverage. If there is only one single scope target
        (e.g., S1 but no S2), we dismiss it.

        :param data: A dataframe with the target data
        :return: A dataframe with the target data sorted by boundary coverage
        """
        # Save the original columns to use when returning the result
        original_columns = data.columns.tolist()

        # Need to save dypes as well since conversion from Series to DataFrame can change them
        original_dtype = data.dtypes
        scope_1_2_mask = data[self.c.COLS.SCOPE].isin([EScope.S1, EScope.S2, EScope.S1S2])

        # Use the boolean mask to filter for Scope 1 and Scope 2
        scope_1_2_data = data[scope_1_2_mask]

        # Group by COMPANY_ID and TIME_FRAME
        grouped_data = scope_1_2_data.groupby(level=[self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME])
                
        def get_best_s1_s2_combination(group: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
            """
            Find the combination of S1 and S2 targets with the highest combined coverage
            :param group: The group of data to analyze
            :return: The row with the highest combined coverage
            """
            group_reset = group.reset_index(drop=True)
            ghg_s1 = self.c.COLS.BASEYEAR_GHG_S1
            ghg_s2 = self.c.COLS.BASEYEAR_GHG_S2

            # Filter the data for S1 and S2 scopes
            idx_s1 = group_reset[group_reset[self.c.COLS.SCOPE] == EScope.S1]
            idx_s2 = group_reset[group_reset[self.c.COLS.SCOPE] == EScope.S2]

            max_coverage = -float('inf')
            best_combination = None

            # Iterate over all combinations of S1 and S2
            for i, row_s1 in idx_s1.iterrows():
                for j, row_s2 in idx_s2.iterrows():
                    # Calculate the weighted coverage
                    weighted_coverage = (
                        (row_s1['coverage_s1'] * row_s1[ghg_s1] + row_s2['coverage_s2'] * row_s2[ghg_s2]) 
                        / (row_s1[ghg_s1] + row_s2[ghg_s2])
                    )

                    # Check if this is the best combination so far
                    if weighted_coverage > max_coverage:
                        max_coverage = weighted_coverage
                        best_combination = (row_s1, row_s2)

            # Combine the results into a single DataFrame
            result = pd.DataFrame([best_combination[0], best_combination[1]]) # type: ignore

            return max_coverage, result
     
        def get_best_combined_s1_s2_coverage(group: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
            """
            Find the combined S1+S2 target with the highest weighted coverage
            :param group: The group of data to analyze
            :return: The row with the highest combined coverage
            """
            ghg_s1 = self.c.COLS.BASEYEAR_GHG_S1
            ghg_s2 = self.c.COLS.BASEYEAR_GHG_S2
            valid_rows = (group[ghg_s1] + group[ghg_s2]) > 0
            filtered_group = group[valid_rows].copy()
            if not filtered_group.empty:
            # Add column weighted_coverage to the DataFrame
                filtered_group.loc[:, 'weighted_coverage'] = (
                    (filtered_group[self.c.COLS.COVERAGE_S1] * filtered_group[ghg_s1] + filtered_group[self.c.COLS.COVERAGE_S2] * filtered_group[ghg_s2])
                    / (filtered_group[ghg_s1] + filtered_group[ghg_s2])
                ) 
                # Find the maximum weighted coverage
                sorted_group = filtered_group.sort_values(by='weighted_coverage', ascending=False)
                max_coverage = sorted_group['weighted_coverage'].max()
                result = sorted_group[sorted_group['weighted_coverage'] == max_coverage].drop('weighted_coverage', axis=1)
            else:
                max_coverage = 0.0
                # If there is no ghg data we can't calculate the combined coverage
                # So we just return the first row
                result = group.iloc[[0]]
            return max_coverage, result
        
        def settle_tied_coverage(single_scope_s1_s2: pd.DataFrame, combined_s1s2: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
            """
            Settle a tie between two targets by selecting the one with the highest ETargetReference
            :param single_scope_s1_s2: A dataframe with two rows representing the S1 and S2 targets
            :param combined_s1s2: A dataframe with the S1+S2 target
            :return: The target with the highest ETargetReference
            """
            # get values of the target types
            single_type_1 = single_scope_s1_s2[self.c.COLS.TARGET_REFERENCE_NUMBER].iloc[0]
            single_type_2 = single_scope_s1_s2[self.c.COLS.TARGET_REFERENCE_NUMBER].iloc[1]
            combined_type = combined_s1s2[self.c.COLS.TARGET_REFERENCE_NUMBER].iloc[0]

            if single_type_1 == single_type_2:
                if single_type_1 > combined_type:
                    return False, single_scope_s1_s2
                elif single_type_1 < combined_type:
                     return True, combined_s1s2
                else: # If both boundary coverage and target type are equal, then we prefer single scope targets
                    return False, single_scope_s1_s2
            else:
                min_single = min(single_type_1, single_type_2)
                if combined_type <= min_single:
                    return True, combined_s1s2
                else:
                    return False, single_scope_s1_s2
                
        # Initialize a list to collect the best entries
        best_entries = []

        # Iterate over each group of company_id and time_frame
        for _, group in grouped_data:
            # Check if both Scope 1 and Scope 2 exist in the group
            s1_data = group[group[self.c.COLS.SCOPE] == EScope.S1]
            s2_data = group[group[self.c.COLS.SCOPE] == EScope.S2]
            combined_s1_s2 = group[group[self.c.COLS.SCOPE] == EScope.S1S2]

            if not s1_data.empty and not s2_data.empty and not combined_s1_s2.empty:
                # If all three scopes exist, we need to compare the combined S1+S2 target with the individual S1 and S2 targets
                # Note that there may be more than one of each of the individual targets
                # Find the row in s1 and s2 with the highest coverage
                single_coverage, best_s1_s2_combination = get_best_s1_s2_combination(pd.concat([s1_data, s2_data]))
                # find the S1+S2 row with the highest weighted coverage
                combined_coverage, best_s1_s2_target = get_best_combined_s1_s2_coverage(combined_s1_s2)
            
                if single_coverage > combined_coverage:
                    best = best_s1_s2_combination
                    combined_best = False
                elif single_coverage < combined_coverage:
                    best = best_s1_s2_target
                    combined_best = True
                else:
                    # If it's a tie we prefer the target with the best target type 
                    combined_best, best = settle_tied_coverage(best_s1_s2_combination, best_s1_s2_target)            
                if combined_best:
                    best = self._split_s1s2_new(best)
                best['to_calculate'] = True
                best_entries.append(best)

            elif not s1_data.empty and not s2_data.empty and combined_s1_s2.empty:
                # If only S1 and S2 exist we find the combination of S1 and S2 with the highest combined coverage
                # Note that there may be more than one of each of the individual targets
                # Find the row in s1 and s2 with the highest coverage
                single_coverage, best_s1_s2_combination = get_best_s1_s2_combination(pd.concat([s1_data, s2_data]))
                #This must be the best combination, so we append it to the best entries
                best_s1_s2_combination['to_calculate'] = True
                best_entries.append(best_s1_s2_combination)

            elif not combined_s1_s2.empty:
                # If only the combined S1+S2 target exists, we find the S1+S2 target with the highest weighted coverage
                # Note that there may be more than one S1+S2 target
                _, best_s1_s2_target = get_best_combined_s1_s2_coverage(combined_s1_s2)
                #This must be the best combination, so we append it to the best entries after splitting
                # Process each row in best_s1_s2_target and apply _split_s1s2_new
                split_rows = []
                for _, row in best_s1_s2_target.iterrows():
                    split_rows.append(self._split_s1s2_new(pd.DataFrame([row])))

                # Concatenate the results
                best_s1_s2_target = pd.concat(split_rows, ignore_index=True)
                best_s1_s2_target['to_calculate'] = True
                best_entries.append(best_s1_s2_target)

            else:
                best_entries.append(group)

        # Concatenate all the best entries
        for df in best_entries:
            if isinstance(df, pd.DataFrame):
                # Restore the original column types for each dataframe in best_entries
                for col in df.columns:
                    if col in original_dtype:
                        df[col] = df[col].astype(original_dtype[col])
        
        result = pd.concat(best_entries, ignore_index=True)
        if isinstance(result, pd.Series):
            result = pd.DataFrame([result.values], columns = original_columns)
        else:
            result = result.reindex(columns=original_columns)
        result = result.set_index([self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE], drop=False)
       
        # Replace the identified rows in the original DataFrame with the sorted DataFrame
        remaining_data = data[~scope_1_2_mask]
        data = pd.concat([result, remaining_data], ignore_index=True)
        # Set the index again to maintain the original structure
       
        data = data.set_index([self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE], drop=False)
            
        return data
