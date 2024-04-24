import librosa
import tempfile
import shutil

def openAudioFile(path: str, sample_rate=48000, offset=0.0, duration=None, fmin=None, fmax=None):
    """Open an audio file.

    Opens an audio file with librosa and the given settings.

    Args:
        path: Path to the audio file.
        sample_rate: The sample rate at which the file should be processed.
        offset: The starting offset.
        duration: Maximum duration of the loaded content.

    Returns:
        Returns the audio time series and the sampling rate.
    """
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type="kaiser_fast")

    return sig, rate

def openCachedFile(filesystem, path, sample_rate=48000, offset=0.0, duration=None):
    
    import tempfile
    import shutil

    bin = filesystem.openbin(path)

    with tempfile.NamedTemporaryFile() as temp:
        shutil.copyfileobj(bin, temp)
        native_sr = librosa.get_samplerate(temp.name)
        sig, rate = openAudioFile(temp.name, sample_rate, offset, duration)

    return sig, rate, native_sr
