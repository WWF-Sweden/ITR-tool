"""Safe summaries for automatic diagnostics; never render validation inputs."""
from pydantic import ValidationError

from ITR.interfaces import IDataProviderCompany, IDataProviderTarget, PortfolioCompany


def validation_error_summary(error: ValidationError) -> str:
    """Return only known schema field names, excluding input values and context."""
    known_fields = (
        IDataProviderCompany.model_fields.keys()
        | IDataProviderTarget.model_fields.keys()
        | PortfolioCompany.model_fields.keys()
    )
    fields = sorted({
        item["loc"][0]
        for item in error.errors(include_input=False, include_context=False, include_url=False)
        if item.get("loc") and item["loc"][0] in known_fields
    })
    if not fields:
        return "missing or invalid data"
    return "missing or invalid field(s): " + ", ".join(fields)
