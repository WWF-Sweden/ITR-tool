# This is target_validation.py
import datetime
import itertools
import logging
import copy

import pandas as pd
from typing import Type, List, Tuple, Optional
from ITR.configs import PortfolioAggregationConfig

from ITR.interfaces import (
    IDataProviderTarget,
    IDataProviderCompany,
    EScope,
    ETimeFrames,
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
        self.s2_targets: List[IDataProviderTarget] = [] #TODO to be removed in production
        self.target_data: pd.DataFrame = pd.DataFrame()
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

        #self.company_data = pd.DataFrame.from_records([c.dict() for c in companies])
        self.group_targets()
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
        target_type = "abs" in target.target_type.lower() or (
            "int" in target.target_type.lower()
            and target.intensity_metric is not None
            and target.intensity_metric.lower() != "other"
        )
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
        # Added in update Oct 22
        target_current = target.end_year >= datetime.datetime.now().year

        # Delete all S1 or S2 targets we can't combine
        # Note that all S1S2S3 pass these tests
        s1 = target.scope != EScope.S1 or (
            not pd.isnull(target.coverage_s1)
            and not pd.isnull(target.base_year_ghg_s1)
            and not pd.isnull(target.base_year_ghg_s2) #TODO - is this correct when looking at individual scopes?
        )
        s2 = target.scope != EScope.S2 or (
            not pd.isnull(target.coverage_s2)
            and not pd.isnull(target.base_year_ghg_s1) #TODO - is this correct when looking at individual scopes?
            and not pd.isnull(target.base_year_ghg_s2)
        )
        s1s2 = target.scope != EScope.S1S2 or (
            target.coverage_s1 == target.coverage_s2 or (
            not pd.isnull(target.base_year_ghg_s1)
            and not pd.isnull(target.base_year_ghg_s2)
            )
        )
        return (
            target_type 
            and target_progress 
            and target_end_year 
            and target_current 
            and s1 
            and s2
            and s1s2
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
        
        return (
            target_end_year
            and target_current
        )

    
    #@staticmethod
    def _split_s1s2s3(self,
        target: IDataProviderTarget,
    ) -> Tuple[IDataProviderTarget, Optional[IDataProviderTarget]]:
        """
        If there is a s1s2s3 scope, split it into two targets with s1s2 and s3

        :param target: The input target
        :return The split targets or the original target and None
        """
        if target.target_type.lower() == "t_score":
            return target, None
        if target.scope == EScope.S1S2S3:
            s1s2, s3 = target.copy(), None
            if (
                not pd.isnull(target.base_year_ghg_s1)
                and not pd.isnull(target.base_year_ghg_s2)
            ) or target.coverage_s1 == target.coverage_s2:
                s1s2.scope = EScope.S1S2
                if (
                    not pd.isnull(target.base_year_ghg_s1)
                    and not pd.isnull(target.base_year_ghg_s2)
                    and target.base_year_ghg_s1 + target.base_year_ghg_s2 != 0
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
            return s1s2, s3
        
        
        else:
            return target, None
        
    #@staticmethod
    def _split_s1s2(self,
        target: IDataProviderTarget,
    ) -> List[IDataProviderTarget]:
        """
        Split the target into two targets, one for the S1 data and one for the S2 data.

        :param target: The target to potentially split.
        :return: A list containing (the original S1S2 target and) 
         the S1 target and the S2 target from the split.
        """
        targets = [target]
        if target.target_type.lower() == "t_score":
            return targets
        # before splitting S1S2 targets we need to verify that there is GHG data to aggregate the scores later
        # TODO - verify that company is one unique row
        company = self.company_data[self.company_data[self.c.COLS.COMPANY_ID] == target.company_id]
        if (not (pd.isnull(company[self.c.COLS.GHG_SCOPE1].item())
            or pd.isnull(company[self.c.COLS.GHG_SCOPE2].item()))
            and target.scope == EScope.S1S2
        ):
            s1 = target.copy()
            s2 = target.copy()
            s1.coverage_s1 = target.coverage_s1
            s2.coverage_s2 = target.coverage_s2
            s1.reduction_ambition = target.reduction_ambition
            s2.reduction_ambition = target.reduction_ambition
            s1.scope = EScope.S1
            s2.scope = EScope.S2
            # Append '_1' and '_2' to the target_ids of s1 and s2, respectively
            s1.target_ids = [id_ + '_1' for id_ in s1.target_ids]
            s2.target_ids = [id_ + '_2' for id_ in s2.target_ids]
            targets.extend([s1, s2])
           
        return targets

    def _combine_s1_s2(self, target: IDataProviderTarget):
        """
        Check if there is an S2 target that matches this target exactly (if this is a S1 target) 
        and combine them into one target.

        :param target: The input target
        :return: The combined target (or the original if no combining was required)
        """
        if target.scope == EScope.S1 and not pd.isnull(target.base_year_ghg_s1):
            matches = [
                t
                for t in self.s2_targets
                if t.company_id == target.company_id
                and t.base_year == target.base_year
                and t.start_year == target.start_year
                and t.end_year == target.end_year
                and t.target_type == target.target_type
                and (
                    'abs' in t.target_type.lower() 
                    or t.intensity_metric == target.intensity_metric
                )
            ]
            if len(matches) > 0:
                matches.sort(key=lambda t: t.coverage_s2, reverse=True)
                s2 = matches[0]
                combined_coverage = (
                    target.coverage_s1 * target.base_year_ghg_s1
                    + s2.coverage_s2 * s2.base_year_ghg_s2
                ) / (target.base_year_ghg_s1 + s2.base_year_ghg_s2)
                target.reduction_ambition = (
                    (
                        target.reduction_ambition
                        * target.coverage_s1
                        * target.base_year_ghg_s1
                        + s2.reduction_ambition * s2.coverage_s2 * s2.base_year_ghg_s2
                    )
                ) / (
                    target.coverage_s1 * target.base_year_ghg_s1
                    + s2.coverage_s2 * s2.base_year_ghg_s2
                )

                target.coverage_s1 = combined_coverage
                target.coverage_s2 = combined_coverage
                # Enforce that we use the combined target - changed 2022-09-01/BBG input
                # Note removed ".value" on 2022-11-23
                target.scope = EScope.S1S2
                # We don't need to delete the S2 target as it'll be definition have a lower coverage than the combined
                # target, therefore it won't be picked for our 9-box grid
                target.target_ids = target.target_ids + s2.target_ids
        return target

    def _cover_s1_s2(self, target: IDataProviderTarget)-> IDataProviderTarget:
        """
        Set the S1 and S2 coverage of a S1+S2 target to the same value.
        :param target: The input target
        :return: The modified target (or the original if no modification was required)
        """
        #TODO - is this method required given the new scaling of reduction ambition?
        if target.scope == EScope.S1S2 and target.coverage_s1 != target.coverage_s2:
            if pd.isna(target.coverage_s1):
                target.coverage_s1 = 0.0
            if pd.isna(target.coverage_s2):
                target.coverage_s2 = 0.0
            combined_coverage = (
                target.coverage_s1 * target.base_year_ghg_s1
                + target.coverage_s2 * target.base_year_ghg_s2
            ) / (target.base_year_ghg_s1 + target.base_year_ghg_s2)
            target.coverage_s1 = combined_coverage
            target.coverage_s2 = combined_coverage
        return target
    
    @staticmethod
    def _convert_s1_s2_into_combined(
        target: IDataProviderTarget,
    ) -> IDataProviderTarget:
        """
        Convert a S1 or S2 target into a S1+S2 target.

        TODO - what is the incidence of targets where base_year_ghg_s2==0
        
        :param target: The input target
        :return: The converted target (or the original if no conversion was required)
        """
        # In both cases the base_year_ghg s1 + s2 should not be zero, else would get ZeroDivisionError
        if target.base_year_ghg_s1 + target.base_year_ghg_s2 != 0:
            if target.scope == EScope.S1:
                coverage = (
                    target.coverage_s1
                    * target.base_year_ghg_s1
                    / (target.base_year_ghg_s1 + target.base_year_ghg_s2)
                )
                target.coverage_s1 = coverage
                target.coverage_s2 = coverage
                target.scope = EScope.S1S2
            elif target.scope == EScope.S2:
                coverage = (
                    target.coverage_s2
                    * target.base_year_ghg_s2
                    / (target.base_year_ghg_s1 + target.base_year_ghg_s2)
                )
                target.coverage_s1 = coverage
                target.coverage_s2 = coverage
                target.scope = EScope.S1S2
        return target

    @staticmethod
    def _scale_reduction_ambition_by_boundary_coverage(
        target: IDataProviderTarget,
        ) -> IDataProviderTarget:
        """
        Change in ITR method 1.5: all targets have their ambition scaled by their boundary coverage.
        :param target: The input target
        :return: The original target with a weighted reduction ambition, if so required
        """
        if target.target_type.lower() == "t_score":
            return target
        elif target.scope == EScope.S1:
            target.reduction_ambition = (
                target.reduction_ambition * target.coverage_s1
            )
        elif target.scope == EScope.S2:
            target.reduction_ambition = (
                target.reduction_ambition * target.coverage_s2
            )
        elif target.scope == EScope.S3:
            target.reduction_ambition = (
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
                #TODO do we need to set these values?
                    target.coverage_s1 = combined_coverage
                    target.coverage_s2 = combined_coverage
                    target.reduction_ambition = (
                        target.reduction_ambition * combined_coverage
                    )
            
        return target
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

        # TODO - what about targets with "0" coverage or "0" base_year_ghg_s2 - breaks the 'combine' logic
        # TODO - this is not needed if we score on separate scopes
        self.s2_targets = list(
            filter(
                lambda target: target.scope == EScope.S2
                and not pd.isnull(target.base_year_ghg_s2)
                and not pd.isnull(target.coverage_s2),
                targets,
            )
        )

        targets = list(
            filter(
                None, itertools.chain.from_iterable(map(self._split_s1s2s3, targets))
            )
        )
        
        targets = list(
            itertools.chain.from_iterable(map(self._split_s1s2, targets))
            )
        
        
        #targets = [self._combine_s1_s2(target) for target in targets]
        targets = [
            self._scale_reduction_ambition_by_boundary_coverage(target)
            for target in targets
        ]
        # Combine S1 and S2 targets that are identical in all but coverage
        # new_targets = []
        # for target in targets:
        #     new_target = self._combine_s1_s2(copy.deepcopy(target))
        #     new_targets.append(target) # keep the original target
        #     if not target.equals(new_target):
        #         new_targets.append(new_target) # add the combined target if it's different
        # targets = new_targets
        # # 
        # targets = [self._cover_s1_s2(target) for target in targets]
        # combined_targets = []
        # for target in targets:
        #     combined_targets.append(target)
        #     new_target = self._convert_s1_s2_into_combined(copy.deepcopy(target))
        #     if not target.equals(new_target):
        #         combined_targets.append(new_target)
        # targets = combined_targets
        #targets = [self._convert_s1_s2_into_combined(target) for target in targets]
        targets = [self._assign_time_frame(target) for target in targets]

        return targets

    def _find_target(self, row: pd.Series, target_columns: List[str]) -> pd.DataFrame:
       
        """
        Find the target that corresponds to a given row. If there are multiple targets available, filter them.

        :param row: The row from the data set that should be looked for
        :param target_columns: The columns to return
        :return: records from the input data, which contains company and target information, that meet specific criteria. For example, record of greatest emissions_in_scope
        """
        self.target_data.sort_index(level=self.target_data.index.names)
        # Find all targets that correspond to the given row
        try:
            target_data = self.target_data.loc[
                (
                    row[self.c.COLS.COMPANY_ID],
                    row[self.c.COLS.TIME_FRAME],
                    row[self.c.COLS.SCOPE],
                )
            ].copy()
            if isinstance(target_data, pd.Series):
                # One match with Target data
                return pd.DataFrame(target_data[target_columns]).T
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
                # Scope 3 targets need to be filtered separately
                if target_data.scope.iloc[0] != EScope.S3:
                    target_data = (
                        target_data.sort_values(
                            by=[
                                self.c.COLS.TARGET_CONFIRM_DATE,
                                coverage_column,
                                self.c.COLS.TARGET_REFERENCE_NUMBER,
                                self.c.COLS.REDUCTION_AMBITION,
                                'END_YEAR_MINUS_BASE_YEAR',
                                self.c.COLS.END_YEAR,
                            ],
                            axis=0,
                            ascending=[False, False, True, False,False, False ],
                        ).iloc[0][target_columns]
                    )
                    result_df = pd.DataFrame([target_data], columns=target_columns)
                    return result_df
                else:
                    target_data = self._find_s3_targets(target_data, target_columns)
                    return target_data
                          
        except KeyError:
            # No target found
            return pd.DataFrame([row], columns=target_columns)
    
    def _find_s3_targets(self, target_data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """
        Find S3 target that correspond to the given row. Note that there may be more
        than one S3 target. We first look for the target with the latest confirmation date.
        Then check if there is more than one target with the same confirmation date.
        The method then returns all targets with the latest confirmation date.

        :param target_data: The target data
        :param target_columns: The columns to return
        :return: The target data that meet the criteria
        """
        target_data['year'] = target_data[self.c.COLS.TARGET_CONFIRM_DATE].dt.year
        latest_year = target_data['year'].max()
        target_data = target_data[target_data['year'] == latest_year]
        final = target_data[target_columns]
   
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
        target_columns = extended_data.columns
       
        results = []
        for _, row in extended_data.iterrows():
            result = self._find_target(row, target_columns)
            results.append(result)
        # Remove empty or all-NA columns from each DataFrame in results (pandas futureproofing)
        results = [df.dropna(how='all', axis=1) for df in results]
        self.data = pd.concat(results, ignore_index=True)

        return self.data
    
   