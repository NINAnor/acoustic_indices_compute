#!/bin/bash

# Load environment variables
source .env

# Files to process
FILE_LIST="files_to_analyze.csv"

# Make sure results and processed files are initialized
touch processed_files_indices.txt
touch analysis_results.csv

# Using GNU Parallel to process each file
cut -d',' -f1 $FILE_LIST | parallel -j 58 python your_script.py {}