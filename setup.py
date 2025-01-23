# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["ITR", "ITR.data"]

package_data = {"": ["*"], "ITR": ["inputs/*"]}

install_requires = [
    "openpyxl==3.1.2",
    "pandas==2.2.2",
    "pydantic==1.10.14",
    "requests==2.32.3",
    "six>=1.16.0",
    "xlrd==2.0.1",
    "xlsxwriter>=3.0.2",
]

setup_kwargs = {
    "name": "WWF-ITR",
    "version": "0.9.4",
    "description": "This package helps companies and financial institutions to assess the temperature alignment of current targets, commitments, and investment and lending portfolios, and to use this information to develop targets for official validation by the SBTi.'",
    "long_description": "> Visit https://wwf-sweden.github.io/ITR-tool/ for the full documentation\n\n> If you have any additional questions or comments send a mail to: ekonomi-finans@wwf.se\n\n# WWF Temperature Scoring\n\nThis package helps companies and financial institutions to assess the temperature alignment of current targets, commitments, and investment and lending portfolios, and to use this information to develop targets for official validation by the SBTi.\n\n## Structure\n\nThe folder structure for this project is as follows:\n\n    .\n    ├── .github                 # Github specific files (Github Actions workflows)\n    ├── docs                    # Documentation files (Sphinx)\n    ├── config                  # Config files for the package\n    ├── ITR                    # The main Python package for the temperature scoring tool\n    └── test                    # Automated unit tests for the ITR package (Nose2 tests)\n\n## Installation\n\nThe ITR package may be installed using PIP. If you'd like to install it locally, use the following command:\n\n```bash\npip install -e .\n```\n\nFor installing the latest stable release from PyPi, run:\n\n```bash\npip install wwf-itr\n```\n\n## Development\n\nTo set up the local development environment with all dependencies, [install poetry](https://python-poetry.org/docs/#installation) and run:\n\n```bash\npoetry install\n```\n\nThis will create a virtual environment inside the project folder under `.venv`.\n\n### Testing\n\nEach class should be unit tested. The unit tests are written using the Nose2 framework. To run the tests, use:\n\n```bash\nnose2 -v\n```\n\n### Publish to PyPi\n\nThe package should be published to PyPi when any changes to the main branch are merged.\n\n**Update package**\n\n1. Bump the version in `pyproject.toml` based on semantic versioning principles.\n2. Run `poetry build`.\n3. Run `poetry publish`.\n4. Check whether the package has been successfully uploaded.\n\n**Initial Setup**\n\n- Create an account on [PyPi](https://pypi.org/).\n",
    "author": "wwf-sweden",
    "author_email": "ekonomi-finans@wwf.se",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.9.7,<4",
}


setup(**setup_kwargs)

 