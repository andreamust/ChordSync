# ChordSync

Code for ChordSync, a conformer-based audio-to-chord synchroniser.
This repository contains the code for the paper "ChordSync: A Conformer-based Audio-to-Chord Synchroniser" by A. Poltronieri, V. Presutti, and M. Rocamora at the 2024 Sound and Music Computing Conference.

The code contains both the code for re-training the model and reproducing the experiments presented in the paper and the code for running the synchronisation on new audio files.

## 🛠️ Environment Setup and Installation

The code is written in Python 3.11 and [PyTorch](https://pytorch.org/) 2.2.0.
We suggest to create a new conda environment to run the code.
To create a new conda environment, run:

```bash
conda create -n chordsync python=3.11
conda activate chordsync
```

To install the required packages, run:

```
pip install -r requirements.txt
```

## 🧠 Training the Model

### 📊 Data Preparation

The model is trained using a subset of the chord annotations included [ChoCo](https://github.com/smahub/ChoCo), the Chord Corpus dataset.
However, the audio files are not included in this repository for copyright reasons.

All the `JAMS` files containing the annotations and the adio files should be placed in two different folders, and the names of the files should match.
For pre-processing the data, run:

```bash
python ChordSync/data_preprocessing.py --audio_folder path/to/audio/folder --jams_folder path/to/jams/folder -max_sequence_length 15 --excerpt_per_song 25 --excerpt_distance 12 --cache_name cache --device cpu --num_workers 4
```

A new folder named `cache` will be created, containing the pre-processed data.
The audio files are pre-processed as mel-spectrograms, as described in the paper. However, it is possible to extract other features by passing the `--feature_type` argument.

### 🏋️‍♂️ Training

To be documented

## 🧪 Testing the Model

To be documented

## 🔁 Running the Synchronisation

To be documented

## 📝 Citation

Paper under review.

## License

MIT License

Copyright (c) 2024 Andrea Poltronieri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
