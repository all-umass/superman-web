# superman-web

A web interface to the superman tools.

To (re)start the server in the background for typical use, run:

    ./restart_server.sh

Use the option `--dry-run` to check what would happen without interfering
with any currently running server.

Or simply run it directly, and handle the details yourself:

    python superman_server.py


### Dependencies

 * `superman` (and its dependencies)
 * `matplotlib >= 1.4.0` (which is higher than the minimum for superman)
 * `tornado`
 * `h5py`
 * `pyyaml`


### Testing

Tests live in the `test/` directory.
Run them directly, or use `python -m unittest discover -s test`.
To generate a nice coverage report:

    coverage run --source server -m unittest discover -s test
    coverage html
