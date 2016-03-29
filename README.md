# superman-server

A web interface to the superman tools.

To (re)start the server in the background for typical use, run:

    ./restart_server.sh

Or simply run it directly, and handle the details yourself:

    python superman_server.py


### Dependencies

 * `superman` (and its dependencies)
 * `matplotlib >= 1.4.0` (which is higher than the minimum for superman)
 * `tornado`
 * `h5py`
