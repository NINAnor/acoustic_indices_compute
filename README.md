# Compute acoustic indices

This repository contains a minimal wrapper around [Maad](https://scikit-maad.github.io/index.html) where the user only has to input a list of files and the script outputs a dataframe of indices for each acoustic file analyzed.

Note that the code makes use of [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) so that the analysis can be done on files from a remote server.

## How to use the script

### Install the dependancies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Change the parameters

List the files you want to analyze in a `files_to_analyze.csv` file in the form:

| filename |
| filecache::ssh://user:password@host/device/2022-10-24T19_36_53.317Z.mp3 |
| filecache::ssh://user:password@host/device/2022-10-24T19_01_36.412Z.mp3 |
| filecache::ssh://user:password@host/device/2022-10-24T11_07_50.161Z.mp3 |

:star: You can change the `gain` or the `sensitivity` parameters in the `.env` file.

### Run the script

```
python3 main.py
```