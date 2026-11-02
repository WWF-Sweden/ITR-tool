Contributing
============

Code, documentation, examples, reproducible bug reports and feature
requests are welcome. Report engine/notebook issues in
`ITR-tool <https://github.com/WWF-Sweden/ITR-tool/issues>`_ and app issues in
`WWF-ITR-APP <https://github.com/WWF-Sweden/WWF-ITR-APP/issues>`_.
Do not include confidential portfolio data in public reports.

Submitting a bug report or a feature request
--------------------------------------------

Include the package version or image tag, Python/OS version where relevant,
steps to reproduce, expected behaviour and error text. Provide a small
non-confidential example and check existing issues first.

Contributing code
-----------------

Use a branch or fork and submit a pull request. Keep user-facing
documentation and examples aligned with behavioural changes. Run checks
appropriate to the change and describe what was verified in the PR.

Getting started
~~~~~~~~~~~~~~~

Use Python 3.10 or later and Poetry. From the ITR-tool repository root:

.. code-block:: console

   poetry install

The development dependencies and optional docs group are defined in
``pyproject.toml``. The configuration uses an in-project virtual environment.
For an editable pip installation of the engine, use
``python -m pip install -e .``; this does not install Poetry's development
or documentation groups.

Coding guidelines
~~~~~~~~~~~~~~~~~

Follow the surrounding code style and include type information where
useful. Tests are in ``test``; use the repository's unittest-compatible
test modules for relevant changes. Do not assume the old workflows
targeting ``master`` run on current ``main`` pull requests; inspect the
actual workflows under ``.github/workflows``.

Releasing
---------

Maintain consistent version values in package metadata and
``ITR/__init__.py``. Build and publish a reviewed release using the
project's release process; merging source alone does not establish that
a new package is available on PyPI. With Poetry, the package commands are:

.. code-block:: console

   poetry build
   poetry publish

After publishing, verify the version on PyPI. The app locks its engine
dependency in its own ``requirements.txt``; its dependency-update PR and
container publication are separate from the Python release. Check both
repositories before describing a release as available to all users.

Generate documentation
~~~~~~~~~~~~~~~~~~~~~~

Edit the Sphinx sources under ``docs`` in ITR-tool. To build locally:

.. code-block:: console

   poetry install --with docs
   poetry run sphinx-build -b html docs docs/_build/html -W --keep-going

Open ``docs/_build/html/index.html`` to review the result. Check navigation,
downloads, code examples and existing public anchors. Generated HTML is
not the source to edit. The ``Docs2Pages`` workflow builds from ``docs``
and deploys to the ``gh-pages`` branch on pushes to ``main`` or manual
dispatch. Changes on a documentation branch do not themselves update
the public site; app-repository changes do not rebuild ITR-tool docs.

Before release, check the documented app startup, notebook and Python
paths against the versions being released. Preserve existing public
page URLs when moving content.

Code of conduct
---------------

Please follow the repository's
`Code of Conduct <https://github.com/WWF-Sweden/ITR-tool/blob/main/CODE_OF_CONDUCT.md>`_.
