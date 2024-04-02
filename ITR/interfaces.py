from enum import Enum
from typing import Optional, Dict, List

import pandas as pd
from pydantic import BaseModel, validator, Field


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
    ghg_s1s2: Optional[float] = 0.0
    ghg_s3: Optional[float] = 0.0

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
    S3 = "S3"
    S1S2 = "S1+S2"
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


class IDataProviderTarget(BaseModel):
    company_id: str
    target_type: str
    intensity_metric: Optional[str]
    scope: EScope
    s3_category: Optional[int]
    coverage_s1: Optional[float]
    coverage_s2: Optional[float]
    coverage_s3: Optional[float]

    reduction_ambition: Optional[float]
    
    base_year: int
    base_year_ghg_s1: Optional[float]
    base_year_ghg_s2: Optional[float]
    base_year_ghg_s3: Optional[float]
    
    start_year: Optional[int]
    end_year: int
    time_frame: Optional[ETimeFrames]
    achieved_reduction: Optional[float] = 0

    target_ids: List[str] = Field(
        default_factory=list,
        description="""Some data providers use a unique identifier for each target. This identifier can then be used to 
        link companies targets to scores. E.g. targets MSCI1235 and MSCI4567 drive a score of 2.5° for Company 123""",
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
    
    @validator("s3_category", pre=True, always=False)
    def validate_f(cls, val):
        if val == "" or val == "nan" or pd.isnull(val):
            return None
        return val

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