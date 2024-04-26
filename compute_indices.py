import logging
import os

import fsspec
import librosa
import maad
import pandas as pd
from dotenv import load_dotenv
from maad import features, sound

from _utils import (add_to_processed_files, get_processed_files, read_df,
                    read_file)

load_dotenv()


def compute_indices(wave, fs, G, S):

    # TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
    #'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
    #'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount'
    # ]

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

    # SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
    #'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
    #'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
    #'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
    #'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
    #'AGI','ROItotal','ROIcover'
    # ]

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


def main(dfpath, G, S):

    processed_path = "processed_files.txt"
    results_file = "analysis_results.csv"

    df = read_df(dfpath)
    processed_files = get_processed_files(processed_path)
    df_indices_all = pd.DataFrame()

    for index, row in df.iterrows():

        filename = row["filename"]
        if filename in processed_files:
            print(f"{filename} has already been analyzed. Skipping.")
            continue

        try:
            wave, fs = read_file(filename)
        except Exception as e:
            df.drop(index, inplace=True)
            print(f"{filename} failed to be analyzed")
            logging.error(f"{filename} failed to be analyzed: {str(e)}")

        df_spec_file = compute_indices(wave, fs, G, S)
        df_spec_file["filename"] = filename
        df_indices_all = pd.concat([df_indices_all, df_spec_file])
        add_to_processed_files(filename, processed_path)

        # Save the DataFrame after processing each file
        df_indices_all.to_csv(results_file, index=False)


if __name__ == "__main__":

    main(
        dfpath=os.getenv("FILES_TO_ANALYZE"),
        G=int(os.getenv("GAIN")),
        S=int(os.getenv("SENSIBILITY")),
    )
