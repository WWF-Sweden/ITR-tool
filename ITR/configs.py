"""
This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""
import os

from ITR.interfaces import ETimeFrames, EScope, S3Category, ETargetReference


class ColumnsConfig:
    # Define a constant for each column used in the input data
    COMPANY_ID = "company_id"
    COMPANY_ISIN = "company_isin"
    COMPANY_LEI = "company_lei"
    COMPANY_ISIC = "isic"
    TARGET_IDS = "target_ids"
    REGRESSION_PARAM = "param"
    REGRESSION_INTERCEPT = "intercept"
    MARKET_CAP = "company_market_cap"
    INVESTMENT_VALUE = "investment_value"
    COMPANY_ENTERPRISE_VALUE = "company_enterprise_value"
    COMPANY_EV_PLUS_CASH = "company_ev_plus_cash"
    COMPANY_TOTAL_ASSETS = "company_total_assets"
    TARGET_REFERENCE_NUMBER = "target_type"
    SCOPE = "scope"
    SCOPE3_CATEGORY = "s3_category"
    AR6 = "ar6"
    REDUCTION_FROM_BASE_YEAR = "reduction_from_base_year"
    START_YEAR = "start_year"
    VARIABLE = "variable"
    SLOPE = "slope"
    TIME_FRAME = "time_frame"
    MODEL = "model"
    ANNUAL_REDUCTION_RATE = "annual_reduction_rate"
    EMISSIONS_IN_SCOPE = "emissions_in_scope"
    TEMPERATURE_SCORE = "temperature_score"
    COMPANY_NAME = "company_name"
    OWNED_EMISSIONS = "owned_emissions"
    COUNTRY = "country"
    SECTOR = "sector"
    GHG_SCOPE1 = "ghg_s1"
    GHG_SCOPE2 = "ghg_s2"
    GHG_SCOPE12 = "ghg_s1s2"
    GHG_SCOPE3 = "ghg_s3"
    COMPANY_REVENUE = "company_revenue"
    CASH_EQUIVALENTS = "company_cash_equivalents"
    TARGET_CLASSIFICATION = "target_classification"
    REDUCTION_AMBITION = "reduction_ambition"
    BASE_YEAR = "base_year"
    END_YEAR = "end_year"
    SBTI_VALIDATED = "sbti_validated"
    TARGET_CONFIRM_DATE = "statement_date"
    ACHIEVED_EMISSIONS = "achieved_reduction"
    ISIC = "isic"
    INDUSTRY_LVL1 = "industry_level_1"
    INDUSTRY_LVL2 = "industry_level_2"
    INDUSTRY_LVL3 = "industry_level_3"
    INDUSTRY_LVL4 = "industry_level_4"
    COVERAGE_S1 = "coverage_s1"
    COVERAGE_S2 = "coverage_s2"
    COVERAGE_S3 = "coverage_s3"
    INTENSITY_METRIC = "intensity_metric"
    INTENSITY_METRIC_AR6 = "intensity_metric"
    TARGET_TYPE_AR6 = "target_type"
    AR6_VARIABLE = "ar6_variable"
    REGRESSION_MODEL = "Regression_model"
    BASEYEAR_GHG_S1 = "base_year_ghg_s1"
    BASEYEAR_GHG_S2 = "base_year_ghg_s2"
    BASEYEAR_GHG_S1S2 = "base_year_ghg_s1s2"
    BASEYEAR_GHG_S3 = "base_year_ghg_s3"
    REGION = "region"
    ENGAGEMENT_TARGET = "engagement_target"
    BASE_YEAR_TS = "base_year_ts"
    END_YEAR_TS = "end_year_ts"
    TO_CALCULATE = "to_calculate"

    # Scope 3 categories - from fundamental data
    GHG_S3_1 = "ghg_s3_1"
    GHG_S3_2 = "ghg_s3_2"
    GHG_S3_3 = "ghg_s3_3"
    GHG_S3_4 = "ghg_s3_4"
    GHG_S3_5 = "ghg_s3_5"
    GHG_S3_6 = "ghg_s3_6"
    GHG_S3_7 = "ghg_s3_7"
    GHG_S3_8 = "ghg_s3_8"
    GHG_S3_9 = "ghg_s3_9"
    GHG_S3_10 = "ghg_s3_10"
    GHG_S3_11 = "ghg_s3_11"
    GHG_S3_12 = "ghg_s3_12"
    GHG_S3_13 = "ghg_s3_13"
    GHG_S3_14 = "ghg_s3_14"
    GHG_S3_15 = "ghg_s3_15"

    # AR6 mapping columns
    PARAM = "param"
    INTERCEPT = "intercept"

    # Output columns
    WEIGHTED_TEMPERATURE_SCORE = "weighted_temperature_score"
    CONTRIBUTION_RELATIVE = "contribution_relative"
    CONTRIBUTION = "contribution"


class PortfolioAggregationConfig:
    COLS = ColumnsConfig


class TemperatureScoreConfig(PortfolioAggregationConfig):

    """
    This factor determines what part of the temperature for a not SBTi-validated company should be the TS and what part
    should be the default score.
    The calculated temperature score should not be lower than the current level of
    global warning which is expressed through the temperature floor constant.
    """

    SBTI_FACTOR = 1
    DEFAULT_SCORE: float = 3.4
    TEMPERATURE_FLOOR: float = 1.5
    TEST_S3_CALCULATION = False # Set to True to print S3 calculation results   
  
    JSON_REGRESSION_MODEL = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "inputs",
        "AR6_regression_model.json",
    )
    MODEL_NUMBER: int = 1
    DEFAULT_INDUSTRY = "Others"

    # VALUE_TARGET_REFERENCE_ABSOLUTE = "absolute"
    # VALUE_TARGET_REFERENCE_T_SCORE = "t_score"
    # VALUE_TARGET_REFERENCE_INTENSITY = "intensity"
    VALUE_TARGET_REFERENCE_INTENSITY_BASE = "inte"

    SLOPE_MAP = {
        ETimeFrames.SHORT: "slopeCA5",
        ETimeFrames.MID: "slopeCA10",
        ETimeFrames.LONG: "slopeCA30",
    }
    #test this
    VALUE_TARGET_REFERENCE = ETargetReference

    INTENSITY_MAPPINGS = {
        ("Revenue", EScope.S1): "INT.emKyoto_gdp",
        ("Revenue", EScope.S2): "INT.emCO2energysupply_SE",
        ("Revenue", EScope.S3): "INT.emKyoto_gdp",
        ("Product", EScope.S1): "INT.emKyoto_gdp",
        ("Product", EScope.S2): "INT.emCO2energysupply_SE",
        ("Product", EScope.S3): "INT.emKyoto_gdp",
        ("Cement", EScope.S1): "INT.emKyoto_gdp",
        ("Cement", EScope.S2): "INT.emCO2energysupply_SE",
        ("Cement", EScope.S3): "INT.emKyoto_gdp",
        ("Oil", EScope.S1): "INT.emKyoto_gdp",
        ("Oil", EScope.S2): "INT.emCO2energysupply_SE",
        ("Oil", EScope.S3): "INT.emKyoto_gdp",
        ("Steel", EScope.S1): "INT.emKyoto_gdp",
        ("Steel", EScope.S2): "INT.emCO2energysupply_SE",
        ("Steel", EScope.S3): "INT.emKyoto_gdp",
        ("Aluminum", EScope.S1): "INT.emKyoto_gdp",
        ("Aluminum", EScope.S2): "INT.emCO2energysupply_SE",
        ("Aluminum", EScope.S3): "INT.emKyoto_gdp",
        ("Power", EScope.S1): "INT.emCO2energysupply_SE",
        ("Power", EScope.S2): "INT.emCO2energysupply_SE",
        ("Power", EScope.S3): "INT.emKyoto_gdp",
        ("Other", EScope.S1): "INT.emKyoto_gdp",
        ("Other", EScope.S2): "INT.emCO2energysupply_SE",
        ("Other", EScope.S3): "INT.emKyoto_gdp",
    }
    intensity_metric_types = set(metric for metric, _ in INTENSITY_MAPPINGS.keys())

    ABSOLUTE_MAPPINGS = {
        # B06: Extraction Of Crude Petroleum And Natural Gas
        ("B06", EScope.S1): "Emissions|Kyoto Gases", 
        ("B06", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("B06", EScope.S3): "Emissions|Kyoto Gases",
        # C23: Manufacture Of Other Non-Metallic Mineral Products ie Cement
        ("C23", EScope.S1): "Emissions|CO2|Energy and Industrial Processes",
        ("C23", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("C23", EScope.S3): "Emissions|Kyoto Gases",
        # C24: Manufacture Of Basic Metals ie Steel, Aluminum
        ("C24", EScope.S1): "Emissions|CO2|Energy and Industrial Processes",
        ("C24", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("C24", EScope.S3): "Emissions|Kyoto Gases",
        # D35: Electricity, Gas, Steam And Air Conditioning Supply
        ("D35", EScope.S1): "Emissions|CO2|Energy|Supply",
        ("D35", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("D35", EScope.S3): "Emissions|Kyoto Gases",
        # H49: Land Transport And Transport Via Pipelines
        ("H49", EScope.S1): "Emissions|Kyoto Gases",
        ("H49", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("H49", EScope.S3): "Emissions|Kyoto Gases",
        # H50: Water Transport
        ("H50", EScope.S1): "Emissions|Kyoto Gases",
        ("H50", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("H50", EScope.S3): "Emissions|Kyoto Gases",
        # H51: Air Transport
        ("H51", EScope.S1): "Emissions|Kyoto Gases",
        ("H51", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("H51", EScope.S3): "Emissions|Kyoto Gases",
        # H52: Warehousing And Support Activities For Transportation
        ("H52", EScope.S1): "Emissions|Kyoto Gases",
        ("H52", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("H52", EScope.S3): "Emissions|Kyoto Gases",
        # H53: Postal And Courier Activities
        ("H53", EScope.S1): "Emissions|Kyoto Gases",
        ("H53", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("H53", EScope.S3): "Emissions|Kyoto Gases",
        ("other", EScope.S1): "Emissions|Kyoto Gases",
        ("other", EScope.S2): "Emissions|CO2|Energy|Supply",
        ("other", EScope.S3): "Emissions|Kyoto Gases",
    }

    TEMPERATURE_RESULTS = "temperature_results"
    INVESTMENT_VALUE = "investment_value"
    
    S3_CATEGORY_MAPPINGS = {
            S3Category.CAT_1: "ghg_s3_1",
            S3Category.CAT_2: "ghg_s3_2",
            S3Category.CAT_3: "ghg_s3_3",
            S3Category.CAT_4: "ghg_s3_4",
            S3Category.CAT_5: "ghg_s3_5",
            S3Category.CAT_6: "ghg_s3_6",
            S3Category.CAT_7: "ghg_s3_7",
            S3Category.CAT_8: "ghg_s3_8",
            S3Category.CAT_9: "ghg_s3_9",
            S3Category.CAT_10: "ghg_s3_10",
            S3Category.CAT_11: "ghg_s3_11",
            S3Category.CAT_12: "ghg_s3_12",
            S3Category.CAT_13: "ghg_s3_13",
            S3Category.CAT_14: "ghg_s3_14",
            S3Category.CAT_15: "ghg_s3_15",
            S3Category.CAT_H_LINE: "ghg_s3"
    }
    EPSILON = 1e-6

class PortfolioCoverageTVPConfig(PortfolioAggregationConfig):
    FILE_TARGETS = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "inputs",
        "current-Companies-Taking-Action.xlsx",
    )
    # To avoid CTA file being downloaded every time and use a local file instead, set USE_LOCAL_CTA = True and set the
    # path to the local file in FILE_TARGETS_CUSTOM_PATH
    FILE_TARGETS_CUSTOM_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "inputs",
        "current-Companies-Taking-Action.xlsx",
    )
    USE_LOCAL_CTA = False
    # If the CTA file is older than a week, the file will be downloaded again
    SKIP_CTA_FILE_IF_EXISTS = True
    # Temporary URL until the SBTi website is updated
    CTA_FILE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTn5UZIBmOxWKNFpmQGWWDczMvdBJ74l2_j0emUH9mxEKylHqh3oMLhu2FXtAV7-bqDxy9Yz_hkWzu8/pub?output=xlsx"
    USE_CUSTOM_FILE_TARGETS_PATH = False
    OUTPUT_TARGET_STATUS = "sbti_target_status"
    OUTPUT_WEIGHTED_TARGET_STATUS = "weighted_sbti_target_status"
    VALUE_STATUS_SET = "Targets set"
    VALUE_STATUS_COMMITTED = "Committed"
    VALUE_STATUS_REMOVED = "Commitment removed"
    
    # VALUE_TARGET_NO = "No target"
    # VALUE_TARGET_SET = "Near-term"
    # VALUE_ACTION_COMMITTED = "Commitment"
    # VALUE_ACTION_TARGET = "Target"

    TARGET_SCORE_MAP = {
        VALUE_STATUS_REMOVED: 0,
        VALUE_STATUS_COMMITTED: 0,
        VALUE_STATUS_SET: 100,
    }

    # SBTi targets overview (TVP coverage)
    COL_COMPANY_NAME = "company_name"
    COL_COMPANY_ISIN = "isin"
    COL_COMPANY_LEI = "lei"
    #COL_ACTION = "Action"
    COL_TARGET = "near_term_status"