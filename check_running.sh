#!/bin/bash

server_pid=$(pgrep -f 'superman_server.py' | head -1)
message=''
if [[ -z "$server_pid" ]]; then
  message="No superman server running"
else
  if curl -s -m 30 -o /dev/null http://nemo.mtholyoke.edu/; then
    exit 0
  fi
  message="Server running, but no response in 30s"
fi
echo "$message at `date`"

superman_dir=$(dirname "${BASH_SOURCE[0]}")
cd "$superman_dir"

yes n | ./restart_server.sh
echo "$message. I noticed it was down at `date`." | \
  mail -s "Superman was down, restarting..." ccarey@cs.umass.edu

