import pandas as pd
import librosa

import maad
from maad import sound, features

def read_df(dfpath):
    return pd.read_csv(dfpath)

def read_file(filepath):
    wave, fs = librosa.load("audio_exemples/2021-09-28T15_59_00.814Z.mp3")
    return wave, fs

def compute_indices(wave, fs, S=-35, G=26+16):
    # Parameters of the audio recorder. This is not a mandatory but it allows
    # to compute the sound pressure level of the audio file (dB SPL) as a
    # sonometer would do.
    #S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
    #G = 26+16       # Amplification gain (26dB (SM4 preamplifier))

    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power,tn,fn,ext = sound.spectrogram (
        x=wave,
        fs=fs,
        window='hann',
        nperseg=1024,
        noverlap=1024//2,
        verbose=False,
        display=False,
        savefig=None
        )

    # Compute the spectral indices:
    df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(
        Sxx_power=Sxx_power,
        tn=tn,
        fn=fn,
        flim_low=[0,1500],
        flim_mid=[1500,8000],
        flim_hi=[8000,20000],
        gain=G,
        sensitivity=S,
        verbose=False,
        display=False)
    
    return df_spec_ind




def main(dfpath):

 
