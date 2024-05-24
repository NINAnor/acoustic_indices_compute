# Compute acoustic indices

This repository contains a minimal wrapper around [Maad](https://scikit-maad.github.io/index.html) where the user only has to input a list of files and the script outputs a dataframe of indices for each acoustic file analyzed.

Note that the code makes use of [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) so that the analysis can be done on files from a remote server.

## How to use the scripts

### Install the dependancies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the script for single files

If you want to obtain the acoustic indices for your files:

```bash
python3 compute_indices.py filecache::ssh://user:password@host/device/2022-10-24T19_36_53.317Z.mp3 
```

If you want to compute the VGGish features for your audio files:

```bash
python3 vggish_embeddings.py filecache::ssh://user:password@host/device/2022-10-24T19_36_53.317Z.mp3 
```

### Run the script for multiple files in parallel

- 1. Install [GNU Parallel](https://www.gnu.org/software/parallel/):

```bash
sudo apt-get install parallel
```

- 2. List the files you want to analyze in a `files_to_analyze.csv` file in the form:

| filename |
|-----------|
| filecache::ssh://user:password@host/device/2022-10-24T19_36_53.317Z.mp3 |
| filecache::ssh://user:password@host/device/2022-10-24T19_01_36.412Z.mp3 |
| filecache::ssh://user:password@host/device/2022-10-24T11_07_50.161Z.mp3 |


- 3. Run the `run_parallel_indices.sh` or the `run_parallel_embeddings.sh`:

```bash
./run_parallel_indices.sh
```

- 4. Get the results

Using the `run_parallel_indices.sh` you should find the results stored in an `output.txt file`

:star: Note if the computation crashes, `GNU parallel` is able to resume the computation where it stopped.
