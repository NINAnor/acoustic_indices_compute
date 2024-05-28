import sys
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from maad import features, sound
import maad
import numpy as np

from _utils import read_file

load_dotenv()

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore")

def compute_indices(wave, fs):

    freq_limits = [1000, 10000]

    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=wave,
        fs=fs,
        window="hann",
        flims=freq_limits,
    )

    Sxx_amplitude, tn, fn, ext = sound.spectrogram(
        x=wave,
        fs=fs,
        window="hann",
        flims=freq_limits,
        mode="amplitude"
    )

    Sxx_noNoise= maad.sound.median_equalizer(Sxx_power) 
    Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)

    # Indices that need Sxx_dB_noNoise
    EVNspFract, EVNspMean, EVNspCount, EVNsp = maad.features.spectral_events(Sxx_dB_noNoise, dt=tn[1]-tn[0], dB_threshold=6, rejectDuration=0, display=False, extent=ext)  
    LFC, MFC, HFC = maad.features.spectral_cover(Sxx_dB_noNoise, fn) 
    ROItotal, ROIcover = maad.features.region_of_interest_index(Sxx_dB_noNoise, tn, fn, display=False, min_roi=164, max_roi=89870, max_ratio_xy=10)

    # Indices that take sqrt(power spectrogram)
    _, _ , ACI  = maad.features.acoustic_complexity_index(Sxx_amplitude)
    BI = maad.features.bioacoustics_index(Sxx_amplitude,fn, flim=(1000, 10000),)
    roughness = maad.features.surface_roughness(Sxx_amplitude, norm='global')

    # Extract the temporal indices:
    df_audio_ind = features.all_temporal_alpha_indices(
        s=wave,
        fs=fs,
        dB_threshold=10,
        rejectDuration=0.01,
        verbose=False,
        display=True,
        Nt = 15000
        )

    EVNtFraction = df_audio_ind.iloc[0]["EVNtFraction"]
    EVNtMean = df_audio_ind.iloc[0]["EVNtMean"]
    EVNtCount = df_audio_ind.iloc[0]["EVNtCount"]
    ACTtCount = df_audio_ind.iloc[0]["ACTtCount"]
    ACTtMean = df_audio_ind.iloc[0]["ACTtMean"]

    return pd.concat([pd.Series(np.mean(EVNsp)),
                      pd.Series(np.median(EVNsp)),
                      pd.Series(np.std(EVNsp)), 
                      pd.Series(np.std(EVNspFract)),
                      pd.Series(np.std(EVNspMean)),
                      pd.Series(np.std(EVNspCount)),
                      pd.Series(MFC), 
                      pd.Series(ROItotal), 
                      pd.Series(ROIcover), 
                      pd.Series(ACI), 
                      pd.Series(BI), 
                      pd.Series(np.mean(roughness)),
                      pd.Series(np.median(roughness)),
                      pd.Series(np.std(roughness)),
                      pd.Series(EVNtFraction),
                      pd.Series(EVNtMean),
                      pd.Series(EVNtFraction),
                      pd.Series(EVNtCount),
                      pd.Series(ACTtCount),
                      pd.Series(ACTtMean)], 
                      axis=1)

def process_file(filename):
    logging.info(f"Processing {filename}")
    wave, fs = read_file(filename)
    df_spec_file = compute_indices(wave, fs)
    df_spec_file["filename"] = filename

    # Append results to the CSV file
    print(df_spec_file.to_csv(header=None, index=False), end='')

if __name__ == "__main__":
    filename = sys.argv[1]
    process_file(filename)