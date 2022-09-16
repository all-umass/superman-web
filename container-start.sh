#!/usr/bin/env bash

function find_server_pid() {
  pgrep -f 'superman_server.py' | head -1
}

nohup python3 superman_server.py &>logs/errors.out &
sleep 1
if [[ -z "$(find_server_pid)" ]]; then
  echo "Error: server died immediately!"
  cat logs/errors.out
  exit 1
fi
