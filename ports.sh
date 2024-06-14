for port in $(seq 3001 4001); do
  nc -z localhost $port > /dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "Port $port is open"
    break
  fi
done