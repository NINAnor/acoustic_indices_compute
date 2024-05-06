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

def process_file(filename, output_dir):
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
    print(f"Embedding saved to {embedding_path}")

def get_processed_files(processed_path):
    try:
        with open(processed_path, "r") as file:
            processed_files = file.read().splitlines()
    except FileNotFoundError:
        processed_files = []
    return processed_files

if __name__ == "__main__":
    # Expect command line arguments: file to process, processed files log, output directory
    filename = sys.argv[1]
    OUTPUT_DIR="embeddings"
    process_file(filename, OUTPUT_DIR)
