Troubleshooting
===============

The app does not open at localhost:8501
---------------------------------------

For Docker, ensure Docker is running and run ``docker compose ps`` and
``docker compose logs -f itr`` from the folder containing your Compose file.
If port 8501 is already occupied, stop the other app or change the host
port to ``127.0.0.1:8502:8501`` and open ``http://localhost:8502``.

If pulling the image fails, check internet access and the
`published container <https://github.com/WWF-Sweden/WWF-ITR-APP/pkgs/container/wwf-itr-app>`_.
An error about a missing platform manifest means that the image does not
provide your platform; use :ref:`app-local-streamlit` as an alternative.

Python cannot import ITR or Streamlit
-------------------------------------

Activate the environment where you installed dependencies. Use
``python -m pip`` to install into that interpreter. In Jupyter, select
the matching kernel and restart it after upgrading an imported package.
The package name is ``wwf-itr`` and the import name is ``ITR``.
For the app, install the app repository's ``requirements.txt``;
installing the scoring package alone does not install Streamlit.

Example files cannot be found
-----------------------------

Run the notebooks from ``examples`` and run resource-download cells when
using Colab. Run ``streamlit run app.py`` from the app repository root.
Use the :doc:`DataRequirements` downloads rather than renaming unrelated
Excel files. Filenames and sheet names must match the instructions.

An uploaded file cannot be read
-------------------------------

Check the sheet names ``fundamental_data`` and ``target_data`` for
target-based provider workbooks. A portfolio needs ``company_name``,
``company_id`` and ``investment_value``. Upload a UTF-8, comma-separated
CSV or use Excel. Keep identifiers as text to preserve leading zeros.
For pre-scored data, use the six required columns and scope/timeframe
values described in :doc:`using_app`.

Scores are missing or use the default score
-------------------------------------------

Check that company IDs match across files and that the selected scopes
and timeframes have relevant targets. Review invalid-target messages,
units, year fields and coverage fractions. A default score means a valid
target-based score was not available for that combination; it is not
evidence of the company's actual future temperature outcome.

For an emissions- or ownership-weighted method, provide the emissions
and financial denominators required by :doc:`DataRequirements`. Do not
substitute zero for unknown values merely to remove an error.

CTA download or portfolio coverage fails
----------------------------------------

Check network access to SBTi or supply a saved CTA Excel file. The file
must have the current columns and recognized statuses listed in
:ref:`sbti-status`. An old ``Action``/``Target`` workbook is not the
current schema. Errors should be investigated rather than interpreted
as zero coverage.

In the app, inspect the CTA source, modification time and fallback
warnings. **Offline mode** prevents downloading; **Upload custom file**
lets you choose an explicit snapshot. The standalone coverage notebook
has no automatic fallback and stops on a failed download.

If coverage is unexpectedly low, check LEI/ISIN quality. LEI is preferred
when available; an unmatched usable LEI does not trigger a second lookup
by ISIN. Unmatched holdings stay in the denominator. Current CTA data
does not reconstruct historical validation status.

Saved data or changed settings seem to disappear
------------------------------------------------

Click **Apply changes** to accept edits and **Save** to persist a dataset.
For Docker, keep the mounted ``data`` folder and run Compose from the
same folder. Cloud storage may be lost on restart. After changing scoring
parameters, click **Run Analysis** again. Downloads are stored by your
browser, separately from the application database.

Report a problem
----------------

Report package/notebook issues in
`ITR-tool <https://github.com/WWF-Sweden/ITR-tool/issues>`_ and interface
or Docker issues in
`WWF-ITR-APP <https://github.com/WWF-Sweden/WWF-ITR-APP/issues>`_. Include
the software version or image tag, operating system, steps to reproduce,
and error text. Use a small non-confidential example rather than a real
portfolio in a public report.
