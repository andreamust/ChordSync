![ChordSync](assets/cs_logo.png)

Code for ChordSync, a conformer-based audio-to-chord synchroniser. The code is based on the paper "ChordSync: A Conformer-based Audio-to-Chord Synchroniser" presented at the 2024 Sound and Music Computing Conference, which is available [here](https://smcnetwork.org/smc2024/papers/SMC2024_paper_id205.pdf).

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
python ChordSync/data/data_preprocessing.py --audio_folder path/to/audio/folder --jams_folder path/to/jams/folder -max_sequence_length 15 --excerpt_per_song 25 --excerpt_distance 12 --cache_name cache --device cpu --num_workers 4
```

A new folder named `cache` will be created, containing the pre-processed data.
The audio files are pre-processed as mel-spectrograms, as described in the paper. However, it is possible to extract other features by passing the `--feature_type` argument.

### 🏋️‍♂️ Training

The model training uses PyTorch Lightning. For training the model using the
parameters described in the paper, run:

```bash
python ChordSync/train.py
```

## 🔁 Running the Synchronisation

To run the synchronisation on a new audio file, run:

```bash
python ChordSync/sync.py --audio_file path/to/audio/file --model_path models/chordsync_v.0.1.0.ckpt --output_folder path/to/output/folder --device cpu
```

## 📝 Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@inproceedings{poltronieri2024chordsync,
    title={ChordSync: A Conformer-based Audio-to-Chord Synchroniser},
    author={Poltronieri, Andrea and Presutti, Valentina and Rocamora, Martín},
    booktitle={Proceedings of the 2024 Sound and Music Computing Conference},
    year={2024},
    month={July},
    location={Porto, Portugal},
    publisher={Sound and Music Computing Network},
    url={https://smcnetwork.org/smc2024/papers/SMC2024_paper_id205.pdf}
}
```

Or use the following plain text citation:

```
A. Poltronieri, V. Presutti, and M. Rocamora, "ChordSync: A Conformer-based Audio-to-Chord Synchroniser," in Proceedings of the 2024 Sound and Music Computing Conference, July 2024, Porto, Portugal. Sound and Music Computing Network.
```

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

```

```
