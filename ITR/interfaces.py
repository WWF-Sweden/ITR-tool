from enum import Enum
from datetime import date
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, Field, ValidationError


class AggregationContribution(BaseModel):
    company_name: str
    company_id: str
    temperature_score: float
    contribution_relative: Optional[float]
    contribution: Optional[float]

    def __getitem__(self, item):
        return getattr(self, item)


class Aggregation(BaseModel):
    score: float
    proportion: float
    contributions: List[AggregationContribution]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregation(BaseModel):
    all: Aggregation
    influence_percentage: float
    grouped: Dict[str, Aggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregationScopes(BaseModel):
    S1: Optional[ScoreAggregation]
    S2: Optional[ScoreAggregation]
    S1S2: Optional[ScoreAggregation]
    S3: Optional[ScoreAggregation]
    S1S2S3: Optional[ScoreAggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregations(BaseModel):
    short: Optional[ScoreAggregationScopes]
    mid: Optional[ScoreAggregationScopes]
    long: Optional[ScoreAggregationScopes]

    def __getitem__(self, item):
        return getattr(self, item)


class ScenarioInterface(BaseModel):
    number: int
    engagement_type: Optional[str] 


class PortfolioCompany(BaseModel):
    company_name: str
    company_id: str
    company_isin: Optional[str]
    company_lei: Optional[str] 
    investment_value: float
    engagement_target: Optional[bool] = False
    user_fields: Optional[dict] = None


class IDataProviderCompany(BaseModel):
    company_name: str
    company_id: str
    isic: str
    ghg_s1: Optional[float] = np.nan
    ghg_s2: Optional[float] = np.nan
    ghg_s1s2: Optional[float] = np.nan 
    ghg_s3: Optional[float] = np.nan

    # Optional fields for scope 3 categories
    ghg_s3_1: Optional[float] = np.nan
    ghg_s3_2: Optional[float] = np.nan
    ghg_s3_3: Optional[float] = np.nan
    ghg_s3_4: Optional[float] = np.nan
    ghg_s3_5: Optional[float] = np.nan
    ghg_s3_6: Optional[float] = np.nan
    ghg_s3_7: Optional[float] = np.nan
    ghg_s3_8: Optional[float] = np.nan
    ghg_s3_9: Optional[float] = np.nan
    ghg_s3_10: Optional[float] = np.nan
    ghg_s3_11: Optional[float] = np.nan
    ghg_s3_12: Optional[float] = np.nan
    ghg_s3_13: Optional[float] = np.nan
    ghg_s3_14: Optional[float] = np.nan
    ghg_s3_15: Optional[float] = np.nan

    country: Optional[str]
    region: Optional[str] 
    sector: Optional[str] 
    industry_level_1: Optional[str] 
    industry_level_2: Optional[str] 
    industry_level_3: Optional[str] 
    industry_level_4: Optional[str] 

    company_revenue: Optional[float] = 0.0
    company_market_cap: Optional[float] = 0.0
    company_enterprise_value: Optional[float] = 0.0
    company_total_assets: Optional[float] = 0.0
    company_cash_equivalents: Optional[float] = 0.0

    sbti_validated: bool = Field(
        False,
        description='True if the SBTi target status is "Target set", false otherwise',
    )
    

class SortableEnum(Enum):
    def __str__(self):
        return self.name

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) >= order.index(other)
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) > order.index(other)
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) <= order.index(other)
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) < order.index(other)
        return NotImplemented


class EScope(SortableEnum):
    S1 = "S1"
    S2 = "S2"
    S1S2 = "S1+S2"
    S3 = "S3"
    S1S2S3 = "S1+S2+S3"

    @classmethod
    def get_result_scopes(cls) -> List["EScope"]:
        """
        Get a list of scopes that should be calculated if the user leaves it open.

        :return: A list of EScope objects
        """
        return [cls.S1S2, cls.S3, cls.S1S2S3]


class ETimeFrames(SortableEnum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"

#test this
class ETargetReference(SortableEnum):
    ABSOLUTE = "absolute"
    INT_TO_ABS = "int_to_abs"
    INTENSITY = "intensity"
    T_SCORE = "t_score"

class S3Category(SortableEnum):
    CAT_1 = 1
    CAT_2 = 2
    CAT_3 = 3
    CAT_4 = 4
    CAT_5 = 5
    CAT_6 = 6
    CAT_7 = 7
    CAT_8 = 8
    CAT_9 = 9
    CAT_10 = 10
    CAT_11 = 11
    CAT_12 = 12
    CAT_13 = 13
    CAT_14 = 14
    CAT_15 = 15
    CAT_H_LINE = 0
    N_A = -1

class IDataProviderTarget(BaseModel):
    company_id: str
    target_type: str
    intensity_metric: Optional[str]
    base_year_ts: Optional[float]
    end_year_ts: Optional[float]
    scope: EScope
    #s3_category: Optional[int]
    s3_category: Optional[S3Category]
    coverage_s1: Optional[float] = np.nan
    coverage_s2: Optional[float] = np.nan
    coverage_s3: Optional[float] = np.nan

    reduction_ambition: Optional[float] = np.nan
    
    base_year: int
    base_year_ghg_s1: Optional[float] = np.nan
    base_year_ghg_s2: Optional[float] = np.nan
    base_year_ghg_s1s2: Optional[float] = np.nan
    base_year_ghg_s3: Optional[float] = np.nan
    
    start_year: Optional[int]
    end_year: int
    statement_date: Optional[date]
    time_frame: Optional[ETimeFrames]
    achieved_reduction: Optional[float] = 0.0
    to_calculate: Optional[bool] = False # Set to True if the target should be calculated

    target_ids: List[str] = Field(
        default_factory=list,
        description="""Some data providers use a unique identifier for each target. This identifier can then be used to 
        link companies targets to scores. E.g. targets MSCI1235 and MSCI4567 drive a score of 2.5° for Company 123""",
    )

    sbti_validated: bool = Field(
        False,
        description='True if the SBTi target status is "Target set", false otherwise',
    )
    
    @classmethod
    def pre(cls, values):
        # Set default values if not provided
        if 'time_frame' not in values:
            values['time_frame'] = None
        return values

    @validator("start_year", pre=True, always=False)
    def validate_e(cls, val):
        if val == "" or val == "nan" or pd.isnull(val):
            return None
        return val
    
    # @validator("s3_category", pre=True, always=False)
    # def validate_f(cls, val):
    #     if val is None:
    #         return None
    #     elif isinstance(val, S3Category):
    #         return val
    #     elif isinstance(val, int):
    #         try:
    #             return S3Category(val)
    #         except ValueError:
    #             raise ValidationError("Invalid value for s3_category")
    #     else:
    #         raise ValidationError("Invalid type for s3_category")
        
    @validator("target_ids", pre=True)
    def convert_to_list(cls, v):
        """
        targets can be combined so target_ids field must be a list
        pre=True is used to ensure that the validator is called before the default_factory
        with pre=True and default_factory the users can supply single strings, e.g. "Target1"
        """
        if isinstance(v, list):
            return v
        if pd.isnull(v):
            return []
        return [v]
    
    def equals(self, other: "IDataProviderTarget") -> bool:
        """
        Check if two targets are equal.
        :param other: The other target
        :return: True if the targets are equal, False otherwise
        """
        return (self.company_id == other.company_id 
                and self.target_type == other.target_type 
                and self.scope == other.scope 
                and self.base_year == other.base_year 
                and self.end_year == other.end_year 
                and self.time_frame == other.time_frame 
                and self.target_ids == other.target_ids
        )