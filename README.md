# superman-web

A web interface to the superman tools.

To (re)start the server in the background for typical use, run:

    ./restart_server.sh

Use the option `--dry-run` to check what would happen without interfering
with any currently running server.

Or simply run it directly, and handle the details yourself:

    python superman_server.py


### Adding a dataset

Datasets are the basic unit of data in the superman server.
Add one by modifying the `datasets.yml` configuration file,
then optionally adding a loader function to the `dataset_loaders.py` module.
Relative paths are evaluated starting from the current working directory
of the process running `superman_server.py`.

### Dependencies

 * `h5py`
 * `matplotlib >= 1.4.0` (which is higher than the minimum for superman)
 * `pyyaml`
 * `superman` (and its dependencies)
 * `tornado`
 * `pandas`

For running tests:

 * `coverage`
 * `mock`


### Testing

Tests live in the `test/` directory.
Run them directly, or use `python -m unittest discover -s test`.
To generate a nice coverage report:

    coverage run --source backend -m unittest discover -s test
    coverage html
