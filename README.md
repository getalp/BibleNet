# BibleNet Pytorch

Implementation in Pytorch of the use case model, as described in the paper "A Large and Clean Multilingual Corpus of Sentence Aligned Spoken Utterances Extracted from the Bible" (accepted at LREC 2020)

## Requirements

- pytorch
- librosa

## Pipeline
### 1) Download the data (or build the corpus yourself using [the following scripts](https://github.com/getalp/mass-dataset))
You will need to download the pre-computed mel-spectrograms of the data set (such a used in the paper's experiments) [here](https://zenodo.org/record/3354711#.XkpfVHVKjmE). These mel-spectrograms were compute with extract_spectrogram.py

### 2) Build the train/val/test splits
Build the train/val/test splits with build_splits.py. This script will need a CSV file as input which sums up which verses are available for which language. This CSV file can be computed with [the following script](https://github.com/getalp/mass-dataset/blob/master/scripts/check-verses.py). 
If you downloaded the pre-computed mel-spectrograms, this file was packed with the mel-spectrograms and is available [here](https://zenodo.org/record/3354711/files/verses.csv)
You may use make_data.sh to build the splits for english-X language pairs.

### 3) Model training and evaluation

Train a model using run_bible.py and evaluate it with test_bible.py

~~~~
python run.py train.json --data-val val.json
~~~~
