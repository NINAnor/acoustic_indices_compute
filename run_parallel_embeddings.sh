#!/bin/bash

export LC_ALL=C

# Load environment variables
source .env

# Files to process
FILE_LIST="files_to_analyze_total.csv"

# Run the script in parallel
parallel --progress --eta --resume --joblog status_embeddings.txt python vggish_embeddings.py :::: $FILE_LIST

# In bash use the following command:
# time systemd-run --scope --user --property=CPUWeight=1 -- sh -c './run_parallel_embeddings.sh'


