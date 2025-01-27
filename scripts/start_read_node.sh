#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

PUBLIC_IP=$(curl -s ifconfig.me)


# Default values if not set in .env
RQLITE_HTTP_ADDR=${RQLITE_HTTP_ADDR:-0.0.0.0:4001}
RQLITE_RAFT_ADDR=${RQLITE_RAFT_ADDR:-0.0.0.0:4002}
RQLITE_HTTP_ADV_ADDR=${RQLITE_HTTP_ADV_ADDR:-$PUBLIC_IP:4001}
RQLITE_RAFT_ADV_ADDR=${RQLITE_RAFT_ADV_ADDR:-$PUBLIC_IP:4002}
RQLITE_DATA_DIR=${RQLITE_DATA_DIR:-$(pwd)/db/}
JOIN_ADDR=${JOIN_ADDR:-}
HOTKEY=${HOTKEY:-}

# Store PID of rqlited
RQLITE_PID=""

# Function to cleanup processes
cleanup() {
    echo -e "\nShutting down rqlited..."
    if [ ! -z "$RQLITE_PID" ]; then
        kill $RQLITE_PID
        wait $RQLITE_PID 2>/dev/null
    fi
    exit 0
}

# Set trap to catch SIGINT (Ctrl+C) and cleanup
trap cleanup SIGINT SIGTERM


# Start RQLite
rqlited -node-id ${HOTKEY} \
  -http-addr ${RQLITE_HTTP_ADDR} \
  -raft-addr ${RQLITE_RAFT_ADDR} \
  -http-adv-addr ${RQLITE_HTTP_ADV_ADDR} \
  -raft-adv-addr ${RQLITE_RAFT_ADV_ADDR} \
  -raft-non-voter=true \
  -join ${JOIN_ADDR} \
  ${RQLITE_DATA_DIR} &

RQLITE_PID=$!

# Wait for the background process
wait $RQLITE_PID
