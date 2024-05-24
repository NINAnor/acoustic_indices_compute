#!/bin/bash

export LC_ALL=C


# Load environment variables
source .env

# Files to process
FILE_LIST="files_to_analyze_total.csv"

# Using GNU Parallel to process each file
parallel --progress --eta --resume --joblog status.txt python compute_indices.py :::: $FILE_LIST

# In bash use the following command:
#time systemd-run --scope --user --property=CPUWeight=1 -- sh -c './run_parallel_indices.sh >> output.txt'
