#!/bin/bash

# Simulation details
OUTPUT_PREFIX="potential"
INTERVAL=1  # Time interval in seconds
OUTPUT_DIRECTORY="$1"
STOP_SIGNAL="${OUTPUT_DIRECTORY}/stop_signal.txt"

EDR_FILES=("em.edr" "nvt.edr" "npt.edr" "md_0_1.edr")  # List of expected EDR files in order

# Function to find the latest available EDR file
function find_latest_edr {
    for file in "${EDR_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "$file"
            return
        fi
    done
}

while true; do
    # Check for the stop signal file
    if [ -f "$STOP_SIGNAL" ]; then
        echo "Stop signal detected. Exiting..."
        break
    fi

    EDR_FILE=$(find_latest_edr)  # Get the latest available EDR file
    FILE_NAME="${OUTPUT_PREFIX}.xvg"

    if [ -f "$FILE_NAME" ]; then
        rm "$FILE_NAME"
    fi

    if [ -z "$EDR_FILE" ]; then
        echo "No EDR file determined. Waiting for file to become available..."
        sleep $INTERVAL
        continue
    fi

    # Run gmx energy to extract the potential energy
    echo "Potential" | gmx energy -f $EDR_FILE -o "${OUTPUT_PREFIX}.xvg"

    # Wait for the next interval
    sleep $INTERVAL
done
