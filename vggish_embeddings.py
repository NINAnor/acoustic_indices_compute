import logging
import os

import fsspec
import librosa
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torchvggish import vggish, vggish_input

from _utils import (add_to_processed_files, get_processed_files, read_df,
                    read_file)

load_dotenv()


def get_embeddings(wave, fs):

    embedding_model = vggish()
    embedding_model.eval()

    example = vggish_input.waveform_to_examples(wave, fs)
    embeddings = embedding_model.forward(example)
    return embeddings


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_embeddings(embedding, path):
    ensure_directory(os.path.dirname(path))  # Ensure the directory exists
    np.save(path, embedding.detach().numpy())  # Save the Numpy array


def main(dfpath):

    processed_path = "processed_files.txt"
    df = read_df(dfpath)
    processed_files = get_processed_files(processed_path)

    for index, row in df.iterrows():

        filename = row["filename"]
        if filename in processed_files:
            print(f"{filename} has already been analyzed. Skipping.")
            continue

        filename = row["filename"]
        try:
            wave, fs = read_file(filename)
        except Exception as e:
            df.drop(index, inplace=True)
            print(f"{filename} failed to be analyzed")
            logging.error(f"{filename} failed to be analyzed: {str(e)}")

        embedding = get_embeddings(wave, fs)
        add_to_processed_files(filename, processed_path)

        # Construct the path for saving embeddings
        embedding_path = filename.replace(
            os.getenv("STRING_TO_REMOVE"), ""
        )  # Remove the unnecessary part
        embedding_path = (
            embedding_path.replace(".mp3", "") + ".npy"
        )  # Change the file extension
        embedding_path = os.path.join(
            os.getenv('PATH_TO_SAVE_EMBEDDINGS'), "embeddings", embedding_path
        )  # Add a root folder for all embeddings

        save_embeddings(embedding, embedding_path)
        add_to_processed_files(filename, processed_path)
        print(f"Embedding saved to {embedding_path}")


if __name__ == "__main__":

    main(
        dfpath=os.getenv("FILES_TO_ANALYZE"),
    )
