import fsspec
import librosa
import pandas as pd


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
