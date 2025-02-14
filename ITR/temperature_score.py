from enum import Enum
from typing import Optional, Tuple, Type, List, Any, cast

import pandas as pd
import numpy as np
import datetime
import os

from .interfaces import (
    ScenarioInterface,
    EScope,
    ETimeFrames,
    ETargetReference,
    Aggregation,
    AggregationContribution,
    ScoreAggregation,
    ScoreAggregationScopes,
    ScoreAggregations,
    PortfolioCompany,
    S3Category,
)
from .portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod
from .configs import TemperatureScoreConfig
from . import data, utils


class ScenarioType(Enum):
    """
    A scenario defines which scenario should be run.
    """

    TARGETS = 1
    APPROVED_TARGETS = 2
    HIGHEST_CONTRIBUTORS = 3
    HIGHEST_CONTRIBUTORS_APPROVED = 4

    @staticmethod
    def from_int(value) -> Optional["ScenarioType"]:
        value_map = {
            1: ScenarioType.TARGETS,
            2: ScenarioType.APPROVED_TARGETS,
            3: ScenarioType.HIGHEST_CONTRIBUTORS,
            4: ScenarioType.HIGHEST_CONTRIBUTORS_APPROVED,
        }
        return value_map.get(value, None)


class EngagementType(Enum):
    """
    An engagement type defines how the companies will be engaged.
    """

    SET_TARGETS = 1
    SET_SBTI_TARGETS = 2

    @staticmethod
    def from_int(value) -> "EngagementType":
        """
        Convert an integer to an engagement type.

        :param value: The value to convert
        :return:
        """
        value_map = {
            0: EngagementType.SET_TARGETS,
            1: EngagementType.SET_SBTI_TARGETS,
        }
        return value_map.get(value, EngagementType.SET_TARGETS)

    @staticmethod
    def from_string(value: Optional[str]) -> "EngagementType":
        """
        Convert a string to an engagement type.

        :param value: The value to convert
        :return:
        """
        if value is None:
            return EngagementType.SET_TARGETS

        value_map = {
            "SET_TARGETS": EngagementType.SET_TARGETS,
            "SET_SBTI_TARGETS": EngagementType.SET_SBTI_TARGETS,
        }
        return value_map.get(value.upper(), EngagementType.SET_TARGETS)


class Scenario:
    """
    A scenario defines the action the portfolio holder will take to improve its temperature score.
    """

    scenario_type: Optional[ScenarioType]
    engagement_type: EngagementType

    def get_score_cap(self) -> float:
        if self.engagement_type == EngagementType.SET_TARGETS:
            return 1.75
        elif (
            self.scenario_type == ScenarioType.APPROVED_TARGETS
            or self.engagement_type == EngagementType.SET_SBTI_TARGETS
        ):
            return 1.5
        else:
            return np.NaN

    def get_default_score(self, default_score: float) -> float:
        if self.scenario_type == ScenarioType.TARGETS:
            return 1.75
        else:
            return default_score

    @staticmethod
    def from_dict(scenario_values: dict) -> Optional["Scenario"]:
        """
        Convert a dictionary to a scenario. The dictionary should have the following keys:

        * number: The scenario type as an integer
        * engagement_type: The engagement type as a string

        :param scenario_values: The dictionary to convert
        :return: A scenario object matching the input values or None, if no scenario could be matched
        """
        scenario = Scenario()
        scenario.scenario_type = ScenarioType.from_int(
            scenario_values.get("number", -1)
        )
        scenario.engagement_type = EngagementType.from_string(
            scenario_values.get("engagement_type", "")
        )

        if scenario.scenario_type is not None:
            return scenario
        else:
            return None

    @staticmethod
    def from_interface(
        scenario_values: Optional[ScenarioInterface],
    ) -> Optional["Scenario"]:
        """
        Convert a scenario interface to a scenario.

        :param scenario_values: The interface model instance to convert
        :return: A scenario object matching the input values or None, if no scenario could be matched
        """
        if scenario_values is None:
            return None

        scenario = Scenario()
        scenario.scenario_type = ScenarioType.from_int(scenario_values.number)
        scenario.engagement_type = EngagementType.from_string(
            scenario_values.engagement_type
        )

        if scenario.scenario_type is not None:
            return scenario
        else:
            return None


class TemperatureScore(PortfolioAggregation):
    """
    This class provides a temperature score based on the climate goals.

    :param default_score: The temp score if a company is not found
    :param model: The regression model to use
    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the TemperatureScoreConfig
                    class and overwriting one of the parameters.
    """

    def __init__(
        self,
        time_frames: List[ETimeFrames],
        scopes: List[EScope],
        default_score: float = TemperatureScoreConfig.DEFAULT_SCORE,
        model: int = TemperatureScoreConfig.MODEL_NUMBER,
        scenario: Optional[Scenario] = None,
        aggregation_method: PortfolioAggregationMethod = PortfolioAggregationMethod.WATS,
        grouping: Optional[List] = None,
        config: Type[TemperatureScoreConfig] = TemperatureScoreConfig,
    ):
        super().__init__(config)
        self.model = model
        self.c: Type[TemperatureScoreConfig] = config
        self.scenario: Optional[Scenario] = scenario
        self.default_score = default_score

        self.time_frames = time_frames
        self.scopes = scopes

        if self.scenario is not None:
            self.default_score = self.scenario.get_default_score(self.default_score)

        self.aggregation_method: PortfolioAggregationMethod = aggregation_method
        self.grouping: list = []
        if grouping is not None:
            self.grouping = grouping

        self.regression_model = pd.read_json(self.c.JSON_REGRESSION_MODEL)
        self.s3_calculation_test = TemperatureScoreConfig.TEST_S3_CALCULATION
        # Save code if we choose to use several models:
        # self.regression_model = self.regression_model[
        #     self.regression_model[self.c.COLS.MODEL] == self.model
        # ]

    def get_target_mapping(self, target: pd.Series) -> Optional[str]:
        """
        Map the target onto an AR6 target (None if not available).

        :param target: The target as a row of a dataframe
        :return: The mapped AR6 target
        """
        if (target[self.c.COLS.SCOPE] == EScope.S1S2
            or target[self.c.COLS.SCOPE] == EScope.S1S2S3
            ):
            map_scope = EScope.S1
        else:
            map_scope = target[self.c.COLS.SCOPE]
        if (
            target[self.c.COLS.TARGET_REFERENCE_NUMBER]
            .strip()
            .lower()
            .startswith(self.c.VALUE_TARGET_REFERENCE_INTENSITY_BASE)
        ):
            return self.c.INTENSITY_MAPPINGS.get(
                (target[self.c.COLS.INTENSITY_METRIC], map_scope), None
            )
        else:
            # Only first 3 characters of ISIC code are relevant for the absolute mappings
            return self.c.ABSOLUTE_MAPPINGS.get(
                (target[self.c.COLS.COMPANY_ISIC][:3], map_scope),
                self.c.ABSOLUTE_MAPPINGS.get(("other", map_scope)),
            )

    def get_annual_reduction_rate(self, target: pd.Series) -> Optional[float]:
        """
        Get the annual reduction rate (or None if not available).
        From version 1.5 the annual reduction rate is calculated as a 
        compund annual reduction rate, CAR.

        :param target: The target as a row of a dataframe
        :return: The annual reduction
        """
        check = pd.isnull(target[self.c.COLS.REDUCTION_AMBITION])
        check = check or pd.isnull(target[self.c.COLS.END_YEAR])
        check = check or pd.isnull(target[self.c.COLS.BASE_YEAR])
        check = check or (target[self.c.COLS.END_YEAR] <= target[self.c.COLS.BASE_YEAR])
        
        if check:
            return None
        elif target[self.c.COLS.REDUCTION_AMBITION] >= 1.0:
            return 1.0
        else:
            CAR = (1-target[self.c.COLS.REDUCTION_AMBITION]) ** float(
                1 / (target[self.c.COLS.END_YEAR] - target[self.c.COLS.BASE_YEAR])
            ) -1
            return CAR

    def get_regression(
        self, target: pd.Series
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the regression parameter and intercept from the model's output.

        :param target: The target as a row of a dataframe
        :return: The regression parameter and intercept
        """
        if pd.isnull(target[self.c.COLS.AR6]):
            return None, None

        regression = self.regression_model[
            (self.regression_model[self.c.COLS.VARIABLE] == target[self.c.COLS.AR6])
            & (
                self.regression_model[self.c.COLS.SLOPE]
                == self.c.SLOPE_MAP[target[self.c.COLS.TIME_FRAME]]
            )
        ]
        if len(regression) == 0:
            return None, None
        elif len(regression) > 1:
            # There should never be more than one potential mapping
            raise ValueError(
                "There is more than one potential regression parameter for this AR6 goal."
            )
        else:
            return (
                regression.iloc[0][self.c.COLS.PARAM],
                regression.iloc[0][self.c.COLS.INTERCEPT],
            )

    def _merge_regression(self, data: pd.DataFrame):
        """
        Merge the data with the regression parameters from the CDP-WWF Warming Function.
        :param data: The data to merge
        :return: The data set, amended with the regression parameters
        """     
        def get_slope(row: pd.Series) -> Optional[str]:
            return self.c.SLOPE_MAP.get(row[self.c.COLS.TIME_FRAME], None)

        data[self.c.COLS.SLOPE] = data.apply(lambda row: get_slope(row), axis=1) # type: ignore

        return pd.merge(
            left=data,
            right=self.regression_model,
            left_on=[self.c.COLS.SLOPE, self.c.COLS.AR6],
            right_on=[self.c.COLS.SLOPE, self.c.COLS.VARIABLE],
            how="left",
        )

    def get_score(self, target: pd.Series) -> Tuple[float, float]:
        """
        Get the temperature score for a certain target based on the annual reduction rate and the regression parameters.

        :param target: The target as a row of a data frame
        :return: The temperature score
        """
   
        if not target.to_calculate:
            return self.default_score, 1.0
        
        # CAR formula won't accept reduction of 100%, so assign the floor temperature score
        if abs(target[self.c.COLS.REDUCTION_AMBITION] - 1.0) < self.c.EPSILON:
            ts = self.c.TEMPERATURE_FLOOR
        # If target is set using CDP-WWF method, use the T_score from equation 5 in method doc
        elif target[self.c.COLS.TARGET_TYPE_AR6].lower() == "t_score":
            ts = max(
                target[self.c.COLS.BASE_YEAR_TS] 
                - (2040 - target[self.c.COLS.BASE_YEAR])
                * ((target[self.c.COLS.BASE_YEAR_TS] - target[self.c.COLS.END_YEAR_TS])
                   / (target[self.c.COLS.END_YEAR] - target[self.c.COLS.BASE_YEAR])),
                self.c.TEMPERATURE_FLOOR
            )
        else:    
            try:       
                ts = max(
                    target[self.c.COLS.REGRESSION_PARAM]
                    * target[self.c.COLS.ANNUAL_REDUCTION_RATE]
                    * 100 * -1      # According to the method doc 
                    + target[self.c.COLS.REGRESSION_INTERCEPT],
                    self.c.TEMPERATURE_FLOOR,
                )
            except TypeError as e:
                print(f"TypeError: {e}, {target[self.c.COLS.REGRESSION_PARAM]}, ambition: {target[self.c.COLS.REDUCTION_AMBITION]}")
                ts = self.default_score
        if target[self.c.COLS.SBTI_VALIDATED]:
            return ts, 0
        else:
            return (
                ts * self.c.SBTI_FACTOR
                + self.default_score * (1 - self.c.SBTI_FACTOR),
                0,
            )

    def aggregate_company_score(
        self, row: pd.Series, company_data: pd.DataFrame
    ) -> Tuple[float, float, list]:
        """
        Get the aggregated temperature score and a temperature result, which indicates how much of the score is 
        based on the default score for a certain company based on the emissions of company.

        :param company_data: The original data, grouped by company, time frame and scope category
        :param row: The row to calculate the temperature score for (if the scope of the row isn't s1s2s3, 
        it will return the original score
        :return: The aggregated temperature score for a company
        """

        if row[self.c.COLS.SCOPE] != EScope.S1S2S3 or (
            row[self.c.COLS.SCOPE] == EScope.S1S2S3
            and row[self.c.COLS.TARGET_TYPE_AR6].lower() == "t_score"
        ):
            return (
                row[self.c.COLS.TEMPERATURE_SCORE], 
                row[self.c.TEMPERATURE_RESULTS],
                row[self.c.COLS.TARGET_IDS],
            )
        
        s1 = company_data.xs(
            (row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S1)
        )
        s2 = company_data.xs(
            (row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S2)
        )
        s1s2 = company_data.xs(
            (row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S1S2)
        )
        s3 = company_data.xs(
            (row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S3)
        )

        s1_ghg = s1[self.c.COLS.GHG_SCOPE1]
        s2_ghg = s2[self.c.COLS.GHG_SCOPE2]
        s1s2_ghg = s1s2[self.c.COLS.GHG_SCOPE12]
        s3_ghg = s3[self.c.COLS.GHG_SCOPE3]

        if s1_ghg > 0 and s2_ghg > 0 and s3_ghg > 0:
            combined_TS = (s1.temperature_score * s1_ghg + 
            s2.temperature_score * s2_ghg + 
            s3.temperature_score * s3_ghg
            ) / (s1_ghg + s2_ghg + s3_ghg)
            combined_TR = (s1.temperature_results * s1_ghg +
            s2.temperature_results * s2_ghg +
            s3.temperature_results * s3_ghg
            ) / (s1_ghg + s2_ghg + s3_ghg)
            combined_targets = list((s1.target_ids or []) + 
            (s2.target_ids or []) + (s3.target_ids or [])) or []
        elif s1s2_ghg > 0 and s3_ghg > 0:
            if s1s2.to_calculate:
                combined_TS = (s1s2.temperature_score * s1s2_ghg + 
                s3.temperature_score * s3_ghg
                ) / (s1s2_ghg + s3_ghg)
            else:
                combined_TS = max(s1s2.temperature_score.item(), s3.temperature_score.item())

            combined_TR = (s1s2.temperature_results * s1s2_ghg + 
            s3.temperature_results * s3_ghg
            ) / (s1s2_ghg + s3_ghg)
            combined_targets = list((s1s2.target_ids or []) + 
            (s3.target_ids or [])) or []
        else:
            combined_TS = max(s1s2.temperature_score.item(), s3.temperature_score.item())
            combined_TR = row[self.c.TEMPERATURE_RESULTS]
            combined_targets = row[self.c.COLS.TARGET_IDS]
       

        return (
            combined_TS,
            combined_TR,
            combined_targets
        ) # type: ignore Pylance doesn't understand that TS and TR are floats
     

    def get_default_score(self, target: pd.Series) -> int:
        """
        Get the temperature score for a certain target based on the annual reduction rate and the regression parameters.

        :param target: The target as a row of a dataframe
        :return: The temperature score
        """
        if (
            pd.isnull(target[self.c.COLS.REGRESSION_PARAM])
            or pd.isnull(target[self.c.COLS.REGRESSION_INTERCEPT])
            or pd.isnull(target[self.c.COLS.ANNUAL_REDUCTION_RATE])
        ):
            return 1
        return 0

    def _prepare_data(self, data: pd.DataFrame):
        """
        Prepare the data such that it can be used to calculate the temperature score.

        :param data: The original data set as a pandas data frame
        :return: The extended data frame
        """
        # IDataProvider provides this optional field as an empty list by default
        # but if call TemperatureScore class with DataFrame directly then it may not be present
        if self.c.COLS.TARGET_IDS not in data.columns:
            data[self.c.COLS.TARGET_IDS] = [[] for _ in range(len(data))]

        # If scope S1S2S3 is in the list of scopes to calculate, we need to calculate the other two as well
        scopes = self.scopes.copy()
        if EScope.S1S2S3 in self.scopes and EScope.S1S2 not in self.scopes:
            scopes.append(EScope.S1S2)
        if EScope.S1S2S3 in scopes and EScope.S3 not in scopes:
            scopes.append(EScope.S3)
        # We also need to calculate scope 1 and scope 2 scores as of version 1.5
        if EScope.S1S2 in scopes and EScope.S1 not in scopes:
            scopes.append(EScope.S1)
            scopes.append(EScope.S2)

        data = data[
            data[self.c.COLS.SCOPE].isin(scopes)
            & data[self.c.COLS.TIME_FRAME].isin(self.time_frames)
        ].copy()

        data[self.c.COLS.TARGET_REFERENCE_NUMBER] = data[
            self.c.COLS.TARGET_REFERENCE_NUMBER
        #].replace({np.nan: self.c.VALUE_TARGET_REFERENCE_ABSOLUTE})
        ].replace({np.nan: ETargetReference.ABSOLUTE.value})
       
        # Replacing NaN with empty list in column target_ids
        data[self.c.COLS.TARGET_IDS] = data[self.c.COLS.TARGET_IDS].apply(
            lambda x: x if isinstance(x, list) else []
        )
        data[self.c.COLS.AR6] = data.apply(
            lambda row: self.get_target_mapping(row), axis=1 # type: ignore
        )  # type: ignore
        data[self.c.COLS.ANNUAL_REDUCTION_RATE] = data.apply(
            lambda row: self.get_annual_reduction_rate(row), axis=1 # type: ignore
        ) # type: ignore
        data = self._merge_regression(data)

        data[self.c.COLS.TEMPERATURE_SCORE], data[self.c.TEMPERATURE_RESULTS] = zip(
            *data.apply(lambda row: self.get_score(row), axis=1)
        )

        data = self.cap_scores(data)
        return data

    def _calculate_company_score(self, data):
        """
        Calculate the combined s1s2s3 scores for all companies.

        :param data: The original data set as a pandas data frame
        :return: The data frame, with an updated s1s2s3 temperature score
        """
     
        company_data = (
            data[
                [
                    self.c.COLS.COMPANY_ID,
                    self.c.COLS.TIME_FRAME,
                    self.c.COLS.SCOPE,
                    self.c.COLS.GHG_SCOPE1,
                    self.c.COLS.GHG_SCOPE2,
                    self.c.COLS.GHG_SCOPE12,
                    self.c.COLS.GHG_SCOPE3,
                    self.c.COLS.TEMPERATURE_SCORE,
                    self.c.COLS.TARGET_IDS,
                    self.c.TEMPERATURE_RESULTS,
                    self.c.COLS.TO_CALCULATE,
                ]
            ]
            .groupby(
                [self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]
            )
            .agg(
                # take the mean of numeric columns, list-append self.c.COLS.TARGET_IDS
                {
                    self.c.COLS.GHG_SCOPE1: "mean",
                    self.c.COLS.GHG_SCOPE2: "mean",
                    self.c.COLS.GHG_SCOPE12: "mean",
                    self.c.COLS.GHG_SCOPE3: "mean",
                    self.c.COLS.TEMPERATURE_SCORE: "mean",
                    self.c.TEMPERATURE_RESULTS: "mean",
                    self.c.COLS.TARGET_IDS: "sum",
                    self.c.COLS.TO_CALCULATE: "any",
                }
            )
        )

        # sum pandas aggregator returns 0 where all input targets were None
        company_data.loc[
            company_data[self.c.COLS.TARGET_IDS] == 0, self.c.COLS.TARGET_IDS
        ] = None

        (
            data[self.c.COLS.TEMPERATURE_SCORE],
            data[self.c.TEMPERATURE_RESULTS],
            data[self.c.COLS.TARGET_IDS],
        ) = zip(
            *data.apply(
                lambda row: self.aggregate_company_score(row, company_data), axis=1
            )
        )
        return data

    def calculate(
        self,
        data: Optional[pd.DataFrame] = None,
        data_providers: Optional[List[data.DataProvider]] = None,
        portfolio: Optional[List[PortfolioCompany]] = None,
    ):
        """
        Calculate the temperature for a dataframe of company data. The columns in the data frame should be a combination
        of IDataProviderTarget and IDataProviderCompany.

        :param data: The data set (or None if the data should be retrieved)
        :param data_providers: A list of DataProvider instances. Optional, only required if data is empty.
        :param portfolio: A list of PortfolioCompany models. Optional, only required if data is empty.
        :return: A data frame containing all relevant information for the targets and companies
        """
        if data is None:
            if data_providers is not None and portfolio is not None:
                data = utils.get_data(data_providers, portfolio)
            else:
                raise ValueError(
                    "You need to pass and either a data set or a list of data providers and companies"
                )

        data = self._prepare_data(data)

        if EScope.S1S2 in self.scopes:
            data = self._calculate_s1s2_score(data)

        # data = self._aggregate_s3_score(data)
        if self.s3_calculation_test:
            s3_data_dump = data[data[self.c.COLS.SCOPE] == EScope.S3].copy()
            # get path to save the data
            path = os.path.join(os.path.dirname(__file__), "../examples/data/local/s3_data_dump.xlsx")
            s3_data_dump.to_excel(path, index=False)
        data = self._aggregate_s3_score(data)

        if EScope.S1S2S3 in self.scopes:
            data = self._calculate_company_score(data)

        # We need to filter the scopes again, because we might have had to add a scope in the preparation step
        data = data[data[self.c.COLS.SCOPE].isin(self.scopes)]
        data.drop(columns=['to_calculate'], inplace=True)
        
        return data

    def _get_aggregations(
        self, data: pd.DataFrame, total_companies: int
    ) -> Tuple[Aggregation, pd.Series, pd.Series]:
        """
        Get the aggregated score over a certain data set. Also calculate the (relative) contribution of each company

        :param data: A data set, containing one row per company
        :return: An aggregated score and the relative and absolute contribution of each company
        """
        data = data.copy()
        weighted_scores = self._calculate_aggregate_score(
            data, self.c.COLS.TEMPERATURE_SCORE, self.aggregation_method
        )
        data[self.c.COLS.CONTRIBUTION_RELATIVE] = weighted_scores / (
            weighted_scores.sum() / 100
        )
        data[self.c.COLS.CONTRIBUTION] = weighted_scores
        contributions = (
            data.sort_values(self.c.COLS.CONTRIBUTION_RELATIVE, ascending=False)
            .where(pd.notnull(data), None)
            .to_dict(orient="records")
        )
        aggregation = Aggregation(
            score=weighted_scores.sum(),
            proportion=len(weighted_scores) / (total_companies / 100.0),
            contributions=[
            AggregationContribution.parse_obj(contribution)
            for contribution in contributions
            ],
        )
        return (
            aggregation,
            data[self.c.COLS.CONTRIBUTION_RELATIVE],
            data[self.c.COLS.CONTRIBUTION],
        )

    def _get_score_aggregation(
        self, data: pd.DataFrame, time_frame: ETimeFrames, scope: EScope
    ) -> Optional[ScoreAggregation]:
        """
        Get a score aggregation for a certain time frame and scope, for the data set as a whole and for the different
        groupings.

        :param data: The whole data set
        :param time_frame: A time frame
        :param scope: A scope
        :return: A score aggregation, containing the aggregations for the whole data set and each individual group
        """
        filtered_data = data[
            (data[self.c.COLS.TIME_FRAME] == time_frame)
            & (data[self.c.COLS.SCOPE] == scope)
        ].copy()
        filtered_data[self.grouping] = filtered_data[self.grouping].fillna("unknown")
        total_companies = len(filtered_data)
        if not filtered_data.empty:
            (
                score_aggregation_all,
                filtered_data[self.c.COLS.CONTRIBUTION_RELATIVE],
                filtered_data[self.c.COLS.CONTRIBUTION],
            ) = self._get_aggregations(filtered_data, total_companies)
            score_aggregation = ScoreAggregation(
                grouped={},
                all=score_aggregation_all,
                influence_percentage=self._calculate_aggregate_score(
                    filtered_data, self.c.TEMPERATURE_RESULTS, self.aggregation_method
                ).sum()
                * 100,
            )

            # If there are grouping column(s) we'll group in pandas and pass the results to the aggregation
            if len(self.grouping) > 0:
                grouped_data = filtered_data.groupby(self.grouping)
                for group_names, group in grouped_data:
                    group_name_joined = (
                        group_names
                        if type(group_names) == str
                        else "-".join([str(group_name) for group_name in group_names])
                    )
                    (
                        score_aggregation.grouped[group_name_joined],
                        _,
                        _,
                    ) = self._get_aggregations(group.copy(), total_companies)
            return score_aggregation
        else:
            return None

    def aggregate_scores(self, data: pd.DataFrame) -> ScoreAggregations:
        """
        Aggregate scores to create a portfolio score per time_frame (short, mid, long).

        :param data: The results of the calculate method
        :return: A weighted temperature score for the portfolio
        """

        score_aggregations = ScoreAggregations() # type: ignore
        for time_frame in self.time_frames:
            score_aggregation_scopes = ScoreAggregationScopes() # type: ignore
            for scope in self.scopes:
                score_aggregation_scopes.__setattr__(
                    scope.name, self._get_score_aggregation(data, time_frame, scope)
                )
            score_aggregations.__setattr__(time_frame.value, score_aggregation_scopes)

        return score_aggregations

    def cap_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the temperature scores in the input data frame to a certain value, based on the scenario that's being used.
        This can either be for the whole data set, or only for the top X contributors.

        :param scores: The data set with the temperature scores
        :return: The input data frame, with capped scores
        """
        if self.scenario is None:
            return scores
              
        if self.scenario.scenario_type == ScenarioType.APPROVED_TARGETS:
            score_based_on_target = ~pd.isnull(
                scores[self.c.COLS.TARGET_REFERENCE_NUMBER]
            )
            scores.loc[
                score_based_on_target, self.c.COLS.TEMPERATURE_SCORE
            ] = scores.loc[score_based_on_target, self.c.COLS.TEMPERATURE_SCORE].apply(
                lambda x: min(x, self.scenario.get_score_cap()) # type: ignore
            )
        elif self.scenario.scenario_type == ScenarioType.HIGHEST_CONTRIBUTORS:
            # Cap scores of 10 highest contributors per time frame-scope combination
     
            aggregations = self.aggregate_scores(scores)
            for time_frame in self.time_frames:
                for scope in self.scopes:
                    number_top_contributors = min(
                        10,
                        len(
                            aggregations[time_frame.value][scope.name].all.contributions
                        ),
                    )
                    for contributor in range(number_top_contributors):
                        company_name = aggregations[time_frame.value][
                            scope.name
                        ].all.contributions[contributor][self.c.COLS.COMPANY_NAME]
                        company_mask = (
                            (scores[self.c.COLS.COMPANY_NAME] == company_name)
                            & (scores[self.c.COLS.SCOPE] == scope)
                            & (scores[self.c.COLS.TIME_FRAME] == time_frame)
                        )
                        scores.loc[
                            company_mask, self.c.COLS.TEMPERATURE_SCORE
                        ] = scores.loc[
                            company_mask, self.c.COLS.TEMPERATURE_SCORE
                        ].apply( # type: ignore
                            lambda x: min(x, self.scenario.get_score_cap()) # type: ignore
                        )
        elif self.scenario.scenario_type == ScenarioType.HIGHEST_CONTRIBUTORS_APPROVED:
            score_based_on_target = scores[self.c.COLS.ENGAGEMENT_TARGET]
            scores.loc[
                score_based_on_target, self.c.COLS.TEMPERATURE_SCORE
            ] = scores.loc[score_based_on_target, self.c.COLS.TEMPERATURE_SCORE].apply(
                lambda x: min(x, self.scenario.get_score_cap()) # type: ignore
            )
        return scores

    def anonymize_data_dump(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize the scores by deleting the company IDs, ISIN and renaming the companies .

        :param scores: The data set with the temperature scores
        :return: The input data frame, anonymized
        """
        scores.drop(
            columns=[self.c.COLS.COMPANY_ID, self.c.COLS.COMPANY_ISIN], inplace=True
        )
        for index, company_name in enumerate(scores[self.c.COLS.COMPANY_NAME].unique()):
            scores.loc[
                scores[self.c.COLS.COMPANY_NAME] == company_name,
                self.c.COLS.COMPANY_NAME,
            ] = "Company" + str(index + 1)
        return scores

    def _calculate_s1s2_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the combined S1S2 score each combination of company_id, time_frame and scope.
        If it was possible på calculate individual S1 and S2 scores, then we calculate the 
        aggregated S1S2 score.

        :param data: The data to calculate the S1S2 score for.
        :return: The S1S2 score.
        """
              
        s1s2_data_mask = data[self.c.COLS.SCOPE].isin([EScope.S1, EScope.S2, EScope.S1S2])
        s1s2_data = data[s1s2_data_mask].copy()
        grouped_data = s1s2_data.groupby(['company_id', 'time_frame'])

        for _, group in grouped_data:
            s1s2_score = np.nan # Initailize the s1s2_score to NaN
            s1_s2_results = 1.0 # Initialize the s1_s2_results to 1.0
            if (group.loc[group['scope'] == EScope.S1,'to_calculate'].any() or
                group.loc[group['scope'] == EScope.S2, 'to_calculate'].any()
            ):
                # S1 and S2 scores are either calulted or default scores from earlier
                s1_score = float(group.loc[group['scope'] == EScope.S1, 'temperature_score'].item())
                s2_score = float(group.loc[group['scope'] == EScope.S2, 'temperature_score'].item())
                s1_results = float(group.loc[group['scope'] == EScope.S1, 'temperature_results'].item())
                s2_results = float(group.loc[group['scope'] == EScope.S2, 'temperature_results'].item())
                s1_ghg = float(group.loc[group['scope'] == EScope.S1, 'ghg_s1'].item())
                s2_ghg = float(group.loc[group['scope'] == EScope.S2, 'ghg_s2'].item())
                # REQUIREMENT 6.5 Temperature Score Aggregation
                
                if not np.isnan(s1_ghg) and not np.isnan(s2_ghg) and s1_ghg != 0 and s2_ghg != 0:
                    try:
                        s1s2_score = (s1_score * s1_ghg + s2_score * s2_ghg) / (s1_ghg + s2_ghg)
                        s1_s2_results = (s1_results * s1_ghg + s2_results * s2_ghg) / (s1_ghg + s2_ghg)
                    except ZeroDivisionError:
                       print("Division by zero")
                       print(f"Target ID {group['target_ids'].values[0]}")
                else:
                    s1s2_score = max(s1_score, s2_score)
                    s1_s2_results = 0.5
                group.loc[group['scope'] == EScope.S1S2, 'temperature_score'] = s1s2_score
                group.loc[group['scope'] == EScope.S1S2, 'temperature_results'] = s1_s2_results
                group.loc[group['scope'] == EScope.S1S2, 'target_type'] = (
                    group.loc[group['scope'] == EScope.S1, 'target_type'].values[0]
                )
                s1_target_ids = group[group['scope'] == EScope.S1]['target_ids'].values 
                s2_target_ids = group[group['scope'] == EScope.S2]['target_ids'].values
                combined_target_ids = list(set().union(*s1_target_ids, *s2_target_ids))
                idx = group.index[group['scope'] == EScope.S1S2][0] 
                group.at[idx, 'target_ids'] = combined_target_ids
                        
                # group = group.infer_objects()
               
                # data.update(group) Works but throws a FutureWarning
               
                group = group.copy() # Important!
                cols_to_update = group.columns.tolist() # Or group.columns if no potential for missing columns
                for index in group.index:
                    if index in data.index: # Essential check to avoid KeyError.
                        data.loc[index, cols_to_update] = group.loc[index, cols_to_update].values #.values is important
   
        return data
          
    def _aggregate_s3_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate Scope 3 category scores into a single score per company and time frame.

        Steps:
        1) Calculate the mean temperature score for `s3_category == CAT_15`.
        2) Remove original CAT_15 rows and append the mean CAT_15 score back to the data.
        3) For each company and time frame:
        a) If GHG data is available for all categories, calculate a weighted average.
        b) If not, calculate a simple average of the scores.
        4) Replace the original scores with the aggregated score.
        """
        # Separate Scope 3 data
        s3_data = data[data['scope'] == EScope.S3].copy()

        # Step 1: Calculate mean temperature score for CAT_15
        # cat_15_data = s3_data[s3_data['s3_category'] == S3Category.CAT_15].copy()
        # mean_cat_15 = cat_15_data.groupby(['company_id', 'time_frame']).agg({
        #     'temperature_score': 'mean',
        #     'target_ids': lambda x: list(set().union(*x)),
        #     'target_type': 'first',
        #     'scope': 'first',
        #     's3_category': 'first',
        #     'ghg_s3_15': 'first',
        #     'company_name': 'first'
        # }).reset_index()
        # Step 1: Calculate mean temperature score for CAT_15
        cat_15_data = s3_data[s3_data['s3_category'] == S3Category.CAT_15].copy()

        # Define the aggregation functions for special columns
        agg_funcs = {
            'temperature_score': 'mean',
            'target_ids': lambda x: list(set().union(*x))
        }

        # Specify group by columns
        group_cols = ['company_id', 'time_frame']

        # Get all other columns that need to use 'first'
        other_cols = [col for col in cat_15_data.columns if col not in group_cols + list(agg_funcs.keys())]

        # Set 'first' as the aggregation function for other columns
        for col in other_cols:
            agg_funcs[col] = 'first'

        # Perform the aggregation
        mean_cat_15 = cat_15_data.groupby(group_cols).agg(agg_funcs).reset_index()


        # Step 2: Remove original CAT_15 rows and append the mean score
        s3_data = s3_data[s3_data['s3_category'] != S3Category.CAT_15]
        s3_data = pd.concat([s3_data, mean_cat_15], ignore_index=True)

        # Ensure all required columns are present
        required_columns = data.columns.tolist()
        for col in required_columns:
            if col not in s3_data.columns:
                s3_data[col] = np.nan
        s3_data = s3_data[required_columns]

        # Step 3: Aggregate scores for each company and time frame
        aggregated_rows = []
        grouped = s3_data.groupby(['company_id', 'time_frame'])
        for _ , group in grouped:
            temperature_scores = np.full(15, 3.4) # Set all scores to the default value
            ghg_columns = [f'ghg_s3_{i}' for i in range(1, 16)]
            ghg_values = group.iloc[0][ghg_columns].values.astype(float)
           
            for _, row in group.iterrows():
                s3_category = row['s3_category']
                if pd.notna(s3_category) and isinstance(s3_category, S3Category):
                    index = s3_category.value - 1  # Convert category to index
                    temperature_scores[index] = row['temperature_score']

            # Calculate weighted or simple average
            if not np.isnan(ghg_values).any().all() and np.nansum(ghg_values) > 0.0:
                weighted_avg = np.sum(ghg_values * temperature_scores) / np.sum(ghg_values)
            else:
                weighted_avg = temperature_scores.mean()

            # Create aggregated row
            aggregated_row = group.iloc[0].copy()
            aggregated_row['temperature_score'] = weighted_avg
            aggregated_row['target_ids'] = sum(group['target_ids'], [])
            if S3Category.CAT_H_LINE in group['s3_category'].values:
                aggregated_row['s3_category'] = S3Category.CAT_H_LINE
            else:
                aggregated_row['s3_category'] = 'Aggregated'

            aggregated_rows.append(aggregated_row)

        # Step 4: Replace original s3_data with aggregated data
        aggregated_data = pd.DataFrame(aggregated_rows)
        data = data[data['scope'] != EScope.S3]
        data = pd.concat([data, aggregated_data], ignore_index=True)
        data.sort_values(by=['company_id', 'time_frame', 'scope'], inplace=True)

        return data