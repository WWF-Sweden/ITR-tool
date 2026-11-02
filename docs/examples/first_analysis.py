"""First analysis with the public WWF ITR sample files."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import ITR
from ITR.data.excel import ExcelProvider
from ITR.interfaces import EScope, ETimeFrames
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
base_url = "https://raw.githubusercontent.com/WWF-Sweden/ITR-tool/main/examples/data/"
for filename in ("data_provider_example_360.xlsx", "example_portfolio_360.csv"):
    path = data_dir / filename
    if not path.exists():
        urlretrieve(base_url + filename, path)

print(f"wwf-itr version: {ITR.__version__}")
provider = ExcelProvider(path=str(data_dir / "data_provider_example_360.xlsx"))
portfolio = pd.read_csv(
    data_dir / "example_portfolio_360.csv",
    dtype={"company_id": str, "company_isin": str, "company_lei": str},
)
companies = ITR.utils.dataframe_to_portfolio(portfolio)
scorer = TemperatureScore(
    time_frames=[ETimeFrames.MID],
    scopes=[EScope.S1, EScope.S2, EScope.S1S2, EScope.S3, EScope.S1S2S3],
    aggregation_method=PortfolioAggregationMethod.WATS,
)
company_scores = scorer.calculate(data_providers=[provider], portfolio=companies)
if scorer.invalid_targets:
    print("Skipped targets:")
    print(pd.DataFrame(scorer.invalid_targets).to_string(index=False))

portfolio_scores = scorer.aggregate_scores(company_scores)
for scope, result in portfolio_scores.model_dump()["mid"].items():
    if result is not None and result["all"] is not None:
        print(f"MID {scope}: {result['all']['score']:.2f} degrees C")
company_scores.to_csv("company_temperature_scores.csv", index=False)
print("Saved company_temperature_scores.csv")
