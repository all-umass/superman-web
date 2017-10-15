# superman-web

A web interface to the superman tools.


## Quick Start

Starting from a fresh download of the source files,
a few steps are required before starting the server for the first time.

### 1: Install Dependencies

Python (2.7 or 3.4+) is the main requirement for running the server.
Several Python packages are needed, available from PyPI via `pip`:

    pip install --use-wheel superman matplotlib tornado pyyaml h5py pandas

If you're not running Linux, `superman` may require special care to install.
See [the superman docs](https://github.com/all-umass/superman#installation) for instructions.

For running tests, you'll want:

    pip install pytest mock coverage


### 2: Configure

An example config file is provided at `config-template.yml`.
Copy this to `config.yml` and edit as needed.
Any values left commented-out or not included will use reasonable defaults.

In the same way, copy `datasets-template.yml` to `datasets.yml`
and update the listings to match your local datasets.


### 3: Add Datasets

Datasets are the basic unit of data in the superman server.
Add one by modifying the `datasets.yml` configuration file,
optionally adding a loader function to the `custom_datasets.py` module.
Relative paths are evaluated starting from the current working directory
of the process running `superman_server.py`,
typically the root of this repository.


### 4: Run

To start (or restart) the server in the background for typical use, run:

    ./restart_server.sh

Use the option `--dry-run` to check what would happen without interfering
with any currently running server.

Or simply run the server directly, and handle the details yourself:

    python superman_server.py

To stop the server without restarting it, use:

    ./restart_server.sh --kill

If you want to verify that everything is working as intended,
try running the test suite (located in the `test/` directory):

    python -m pytest

To generate a nice code coverage report:

    coverage run --source backend -m pytest
    coverage html
