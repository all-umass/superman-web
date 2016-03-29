#!/usr/bin/env bash

# Parse result of geoiplookup to fit concisely on a line.
function ip_info() {
  geoiplookup $1 | sed 1d | cut -d: -f2 | cut -d' ' -f3- | cut -d, -f2,3 | xargs
}
export -f ip_info

function start_server() {
  echo "Starting new server..."
  nohup python superman_server.py &>logs/errors.out &
  echo "Use 'tail -f logs/server.log' to check on it"
}

# Use gawk if available, otherwise use awk
AWK=$(gawk -V >/dev/null 2>&1 && echo "gawk" || echo "awk")

# if --dry-run is passed, don't kill or start anything
dry_run=false
if [[ $1 = "--dry-run" ]]; then
  dry_run=true
fi

# Check if the webserver is still running
server_pid=$(pgrep -f 'superman_server.py' | head -1)
if [[ -z "$server_pid" ]]; then
  echo "No currently running server found."
  $dry_run || start_server
  exit
fi

# Grep log entries for successful GET or POST requests,
#  convert times to seconds before now (deltas),
#  find the last delta for each unique IP addreess,
#  then print a table of info for each one in the last n minutes.
$AWK '
BEGIN {
  now = systime();
}
/200 (GET|POST)/ {
  tstamp = ($1 " " $2);
  ip = $7;
  gsub(/[)(]/, "", ip);
  gsub(/[[\]:-]/, " ", tstamp);
  gsub(/,[0-9]+/, "", tstamp);
  d = now - mktime(tstamp);
  print d, ip
}' <logs/server.log \
 | sort -n | sort -u -k2,2 | sort -rn | $AWK '
{
  mins = sprintf("%8.2f", $1/60);
  ("/usr/bin/env bash -c \"ip_info " $2 "\"") | getline info;
  lines = (lines mins " | " sprintf("%15s",$2) " | " info "\n");
}
END {
  if (lines) {
    print "\nRecent webserver activity:";
    print "Mins ago |   IP address    |    GeoIP info";
    print "------------------------------------------------------";
    print lines;
    if ((mins + 0) < 30) {
      exit 1
    }
  }
}'

# If there were any recent entries, ask the user to proceed.
if [[ $? -ne 0 ]]; then
  $dry_run && exit
  read -n1 -s -p "Proceed? [y/N] " response
  echo
  if [[ $response != "y" ]]; then
    exit 1;
  fi
fi

$dry_run && exit
echo "Killing old server (pid: $server_pid)"
while [[ -n "$server_pid" ]]; do
  kill "$server_pid"
  server_pid=$(pgrep -f 'superman_server.py' | head -1)
done

start_server

