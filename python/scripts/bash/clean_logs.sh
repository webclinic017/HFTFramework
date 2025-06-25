#!/bin/bash

# Check if LAMBDA_LOGS_PATH environment variable is set
if [ -z "$LAMBDA_LOGS_PATH" ]; then
    echo "Error: LAMBDA_LOGS_PATH environment variable is not set."
    exit 1
fi

# Navigate to the log directory
cd "$LAMBDA_LOGS_PATH" || exit

# Find and delete files older than a week
find . -name "*.log" -type f -mtime +10 -exec rm -f {} \;

# Find and delete empty directories
find . -depth -type d -empty -delete

echo "Old log files and empty folders deleted successfully."