import fsspec
import librosa
import pandas as pd
import tempfile
import shutil
import os


def read_df(dfpath):
    return pd.read_csv(dfpath)


def read_file(filepath):
    with fsspec.open(filepath) as f:
        wave, fs = librosa.load(f)
    return wave, fs


def get_processed_files(processed_path):
    try:
        with open(processed_path, "r") as file:
            processed_files = file.read().splitlines()
    except FileNotFoundError:
        processed_files = []
    return processed_files


def add_to_processed_files(filename, processed_path):
    with open(processed_path, "a") as file:
        file.write(filename + "\n")


def clean_tmp(directory="/tmp"):
    # Target only directories starting with 'tmp'
    for folder_name in os.listdir(directory):
        if folder_name.startswith("tmp"):
            folder_path = os.path.join(directory, folder_name)
            try:
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
            except Exception as e:
                print(f'Failed to delete {folder_path}. Reason: {e}')

