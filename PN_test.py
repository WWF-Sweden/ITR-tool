import ITR
from ITR.data.excel import ExcelProvider
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.portfolio_coverage_tvp import PortfolioCoverageTVP
from ITR.temperature_score import TemperatureScore, Scenario, ScenarioType
from ITR.target_validation import TargetProtocol
from ITR.interfaces import ETimeFrames, EScope
from examples.utils import print_aggregations
import pandas as pd

provider = ExcelProvider(path="/Users/peternystrom/WWF/ITR-tool/examples/data/data_provider_example.xlsx")
df_portfolio = pd.read_csv("/Users/peternystrom/WWF/ITR-tool/examples/data/example_portfolio.csv", encoding="iso-8859-1")
companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
"""
scenario = Scenario()
scenario.scenario_type = ScenarioType.APPROVED_TARGETS
scenario.engagement_type = None
scenario.aggregation_method = PortfolioAggregationMethod.WATS
scenario.grouping = None
"""


temperature_score = TemperatureScore(                  # all available options:
    time_frames=list(ITR.interfaces.ETimeFrames),     # ETimeFrames: SHORT MID and LONG
    #time_frames=[ETimeFrames.SHORT],     # ETimeFrames: SHORT MID and LONG
    scopes=[EScope.S1, EScope.S2, EScope.S1S2, EScope.S3, EScope.S1S2S3],    # EScopes: S3, S1S2 and S1S2S3
    #scopes=[EScope.S1S2],    # EScopes: S3, S1S2 and S1S2S3
    aggregation_method=PortfolioAggregationMethod.WATS, # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS.
    #scenario=scenario
    #fallback_score=4.1,
)
amended_portfolio = temperature_score.calculate(data_providers=[provider], portfolio=companies)
# aggregated_scores = temperature_score.aggregate_scores(amended_portfolio) 
#pd.DataFrame(aggregated_scores.dict()).applymap(lambda x: round(x['all']['score'], 2))
# pd.DataFrame(aggregated_scores.dict()).applymap(lambda x: round(x['all']['score'], 2) 
#             if x is not None and x['all'] is not None and 'score' in x['all'] else None)

aggregated_portfolio = temperature_score.aggregate_scores(amended_portfolio)
print_aggregations(aggregated_portfolio)
#amended_portfolio.to_csv("/home/mountainrambler/ITR/ITR-tool/examples/data/local/results_z.csv") 