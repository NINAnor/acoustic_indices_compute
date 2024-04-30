import sys
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from maad import features, sound

from _utils import add_to_processed_files, get_processed_files, read_file

load_dotenv()

def compute_indices(wave, fs, G, S):

    # Code borrowed from: https://scikit-maad.github.io/_auto_examples/2_advanced/plot_extract_alpha_indices.html

    # Compute the temporal indices
    df_audio_ind = features.all_temporal_alpha_indices(
        s=wave,
        fs=fs,
        gain=G,
        sensibility=S,
        dB_threshold=3,
        rejectDuration=0.01,
        verbose=False,
        display=False,
    )

    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=wave,
        fs=fs,
        window="hann",
        nperseg=1024,
        noverlap=1024 // 2,
        verbose=False,
        display=False,
        savefig=None,
    )

    # Compute the spectral indices:
    df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(
        Sxx_power=Sxx_power,
        tn=tn,
        fn=fn,
        flim_low=[0, 1500],
        flim_mid=[1500, 8000],
        flim_hi=[8000, 20000],
        gain=G,
        sensitivity=S,
        verbose=False,
        display=False,
    )

    return pd.concat([df_spec_ind, df_audio_ind], axis=1)

def process_file(filename, G, S, processed_path, results_file):
    if filename in get_processed_files(processed_path):
        print(f"{filename} has already been analyzed. Skipping.")
        return

    print(f"Processing {filename}")
    wave, fs = read_file(filename)
    df_spec_file = compute_indices(wave, fs, G, S)
    df_spec_file["filename"] = filename

    # Append results to the CSV file
    df_spec_file.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    add_to_processed_files(filename, processed_path)

if __name__ == "__main__":
    filename = sys.argv[1]
    G = int(os.getenv("GAIN"))
    S = int(os.getenv("SENSIBILITY"))
    processed_path = "processed_files_indices.txt"
    results_file = "analysis_results.csv"

    process_file(filename, G, S, processed_path, results_file)