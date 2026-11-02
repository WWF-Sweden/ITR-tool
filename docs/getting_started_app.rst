Getting Started with the App
============================

The WWF ITR App is a Streamlit interface to the ``wwf-itr`` calculation
engine. Use it to load data, calculate temperature scores, explore
companies and scenarios, and download results without writing analysis code.

Try the online demo
-------------------

Open the `WWF ITR App <https://wwf-itr.streamlit.app/>`_ in your browser.
Choose **Sample Data**, click **Load Data**, and follow :doc:`using_app`.
No installation is required.

The public demo runs on Streamlit Community Cloud. Use sample or public
data there; do not upload confidential or proprietary portfolio data.
For your own confidential data, run the app locally using Docker or Python.
Cloud storage is temporary across app restarts; download any results you
want to keep.

.. _app-docker:

Run locally with Docker
-----------------------

Docker runs the same Streamlit app on your own computer. You do not need
to install Python or clone either GitHub repository.

1. Install and start Docker. See the official instructions for
   `Windows <https://docs.docker.com/desktop/setup/install/windows-install/>`_,
   `macOS <https://docs.docker.com/desktop/setup/install/mac-install/>`_, or
   `Linux <https://docs.docker.com/engine/install/>`_. Docker Desktop
   includes Compose; on Linux, also install the
   `Compose plugin <https://docs.docker.com/compose/install/linux/>`_.
2. Create a folder for the app, for example ``wwf-itr-app``. Download
   :download:`docker-compose.yml <docker-compose.yml>` into that folder.
   The file contains:

   .. literalinclude:: docker-compose.yml
      :language: yaml

3. Open a terminal in that folder and start the app:

   .. code-block:: console

      docker compose up -d --pull always

4. Open `http://localhost:8501 <http://localhost:8501>`_ and follow
   :doc:`using_app`.

The port is bound to ``127.0.0.1`` for access from your own computer.
The first start downloads the image from GitHub Container Registry.
Example data may also be downloaded from GitHub when not already present.
SBTi data is downloaded when required unless you select an uploaded CTA
file or use the app's offline option with available local data.

Stop, restart and update
~~~~~~~~~~~~~~~~~~~~~~~~

Run these commands from the folder containing ``docker-compose.yml``.
To stop the app and remove its container:

.. code-block:: console

   docker compose down

To start it again and check for a newer published image:

.. code-block:: console

   docker compose up -d --pull always

To inspect its status and logs:

.. code-block:: console

   docker compose ps
   docker compose logs -f itr

Press ``Ctrl+C`` to stop following the logs. ``latest`` follows the latest
published app image. For a reproducible installation, use an existing
release tag or image digest from the
`container registry <https://github.com/WWF-Sweden/WWF-ITR-APP/pkgs/container/wwf-itr-app>`_
and record it with the analysis. App image tags and Python package versions
are separate version identifiers.

Keep your saved data
~~~~~~~~~~~~~~~~~~~~

The volume ``./data:/app/data`` keeps saved application data in the
``data`` folder beside your Compose file:

* ``data/itr_data.db`` contains datasets explicitly saved using **Save**.
* ``data/cta/current-Companies-Taking-Action.xlsx`` is the app's CTA cache.

These files survive ``docker compose down`` and image updates. Back up
the folder before updating and keep it if you want to retain saved datasets.
Loading or editing a dataset does not automatically save it to the database.
Temporary uploads are not all retained by the volume. Downloaded exports
are saved through your browser, normally to your Downloads folder.

.. _app-local-streamlit:

Run locally with Python and Streamlit
-------------------------------------

This alternative is useful if you want to work with the app's source code.
Use Python 3.11, the version used by the app's Dockerfile and test workflow,
and Git. Clone the app repository:

.. code-block:: console

   git clone https://github.com/WWF-Sweden/WWF-ITR-APP.git
   cd WWF-ITR-APP

Create and activate a virtual environment. On macOS or Linux:

.. code-block:: console

   python3.11 -m venv .venv
   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1

Then, from the app repository root:

.. code-block:: console

   python -m pip install -r requirements.txt
   python -m streamlit run app.py

Open ``http://localhost:8501``. Stop the server with ``Ctrl+C``.
The app's pinned requirements install its ``wwf-itr`` dependency.
Installing ``wwf-itr`` alone does not install the Streamlit app.

For input files and the first analysis, continue with :doc:`using_app`.
If startup fails, see :doc:`troubleshooting`.
