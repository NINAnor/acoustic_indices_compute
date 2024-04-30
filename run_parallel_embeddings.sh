#!/bin/bash

# File containing the list of files to process
FILE_LIST="files_to_analyze.csv"
PROCESSED_FILES="processed_files.txt"
OUTPUT_DIR="embeddings"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Read filenames from a CSV and run them through the Python script in parallel
cat $FILE_LIST | parallel --bar -j 58 python vggish_embeddings.py {} $PROCESSED_FILES $OUTPUT_DIR
