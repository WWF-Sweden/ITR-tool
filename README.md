# ITR-tool
WWF version of the SBTi-Finance-tool, based on the CDP-WWF Temperature rating methodology

> Visit https://wwf-sweden.github.io/ITR-tool/ for the full documentation

> If you have any additional questions or comments send a mail to: ekonomi-finans@wwf.se

## About the tool

This package helps companies and financial institutions to assess the temperature alignment of current
targets, commitments, and investment and lending portfolios, and to use this information to develop
targets for official validation by the SBTi.

The WWF Finance toolkit can be used in different ways:

- Users can integrate the Python package in their codebase
- Using Notebooks, either locally or on Google Colab

## Structure

The folder structure for this project is as follows:

    .
    ├── .github                 # Github specific files (Github Actions workflows)
    ├── docs                    # Documentation files (Sphinx)
    ├── config                  # Config files for the Docker container
    ├── ITR                     # The main Python package for the temperature alignment tool
    └── test                    # Automated unit tests for the SBTi package (Nose2 tests)

## Installation

The ITR package may be installed using PIP. If you'd like to install it locally use the following command. For testing or production please see the deployment section for further instructions

```bash
pip install -e .
```

For installing the latest stable release in PyPi run:

```bash
pip install wwf-itr
```

## Development

To set up the local dev environment with all dependencies, [install poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions) and run

```bash
poetry install
```

This will create a virtual environment inside the project folder under `.venv`.

### Testing

Each class should be unit tested. The unit tests are written using the Nose2 framework.
The setup.py script should have already installed Nose2, so now you may run the tests as follows:

```bash
nose2 -v
```

### Publish to PyPi

The package should be published to PyPi when any changes to main are merged.

Update package

1. bump version in `pyproject.toml` based on semantic versioning principles
2. run `poetry build`
3. run `poetry publish`
4. check whether package has been successfully uploaded

**Initial Setup**

- Create account on [PyPi](https://pypi.org/)

