import sys
import os
import tempfile
import librosa
import numpy as np
from torchvggish import vggish, vggish_input
import pandas as pd

from _utils import read_file

from dotenv import load_dotenv
load_dotenv()

def get_embeddings(wave, fs):
    embedding_model = vggish()
    embedding_model.eval()
    example = vggish_input.waveform_to_examples(wave, fs)
    embeddings = embedding_model.forward(example)
    return embeddings

def save_embeddings(embedding, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embedding.detach().numpy())  # Save the Numpy array

def process_file(filename, processed_path, output_dir):
    # Check if already processed
    if filename in get_processed_files(processed_path):
        print(f"{filename} has already been analyzed. Skipping.")
        return
    print(filename)
    # Read audio file
    wave, fs = read_file(filename)  # sr=None to preserve original sampling rate

    # Compute embeddings
    embedding = get_embeddings(wave, fs)

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

    # Mark as processed
    add_to_processed_files(filename, processed_path)
    print(f"Embedding saved to {embedding_path}")

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

if __name__ == "__main__":
    # Expect command line arguments: file to process, processed files log, output directory
    filename = sys.argv[1]
    processed_files_path = sys.argv[2]
    output_directory = sys.argv[3]
    process_file(filename, processed_files_path, output_directory)
