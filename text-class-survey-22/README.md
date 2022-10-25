# Text Classification survey

Code used for experiments in papers:

- Gasparetto A, Marcuzzo M, Zangari A, Albarelli A. (2022) A Survey on Text Classification Algorithms: From Text to Predictions. _Information_ 13, no. 2: 83. https://doi.org/10.3390/info13020083
- Gasparetto A, Zangari A, Marcuzzo M, Albarelli A. (2022) A survey on text classification: Practical perspectives on the Italian language. _PLOS ONE_ 17(7): e0270904. https://doi.org/10.1371/journal.pone.0270904

Includes reproduction instructions for

- Classical methods (SVM, NB);
- Neural based methods (FastText, XMLCNN, BiLSTMs);
- Transformer based methods.

## Installation

This procedure has been tested on Ubuntu 20.04.3 LTS with Python 3.8.11. Experiments were run on Nvidia GeForce RTX 2080
Ti with CUDA 10.1.

Install using conda:

```
conda env create -f environment.yml
```

Activate environment:

```
conda activate experiments-tc
```

## Reproducing results

The RCV1En and RCV2It/Fr datasets are derived from RCV1 and RCV2, respectively. These are only available after a request
to NIST, and cannot be disclosed without their consent. Hence, we include [below](#rcv1---rcv2-news-categorization)
instructions on how to generate these datasets after obtaining the RCV1/2 from NIST.

In the case of Wikipedia articles, datasets are provided in a Google Drive folder.

1. Download the datasets
   from [Zenodo](https://doi.org/10.5281/zenodo.7244893);
2. Extract them inside folder `data/raw/` (e.g. `data/raw/enwiki/` contains the EnWiki-100 dataset);
    1. This can be achieved using `tar -xzf {file}.tar.gz -C {destination_folder}`.
3. Download pretrained embeddings as described [below](#download-embeddings);
4. Generate training and testing splits as described [below](#dataset-generation) (only step 5 and 3 for Wikipedia and
   RCV* respectively);
    1. The provided Wikipedia datasets include the list of 100 topics we used. To change the number of topics see step 4.
5. Train models and test them as described [here](#usage-example).

## Dataset generation

This section describes how to generate datasets from raw Wikipedia dumps and RCV1/2 datasets.

### Wikipedia dataset (topic labelling)

1. Download datasets for each language (IT/FR/EN) from here: https://dumps.wikimedia.org/backup-index.html;
2. Download archives named `{lang}wiki-{date}-pages-articles.xml.bz2` and place each of them wherever you prefer (we
   place them in `data/raw/{lang}wiki`);
    - Substitute `{lang}` with the appropriate language prefix (e.g. `it` for Italian).
3. Run `python utils/wikiextractor/WikiExtractor.py` providing the path to the source dump and the output folder where
   articles and topics will be extracted;
    - E.g. `python utils/wikiextractor/WikiExtractor.py {dump_archive_path} --json -o {output_folder}`;
    - We use `data/raw/{lang}wiki/extracted` as output folder.
4. (Optional) You may choose to define the topics to be kept, which defaults to 100;
    1. You may run `utils/visualization.py` to obtain a bar plot of article/topic frequency;
    2. You may then set appropriate parameters to adjust as desired.
5.

**(Option a).** Generate an individual split using `python src/datasets/generation/tl/{language}wiki.py`.

**(Option b).**  Generate splits using `python src/datasets/generation/generate_all.py`. To speed up the
process, this file expects all datasets to be present; if this is not the case, manual editing is required.
By default, this will create 4 stratified splits for each dump and limit the maximum number of
documents per label to 50000. **This will also create RCV1/2 splits**.

After these steps the generated datasets will appear in the `/data/generated` folder with the following structure:

```
.
├── enwiki-1            // split 1
│   ├── encoder.bin.xz
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── enwiki-2            // split 2
│   ├── encoder.bin.xz
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
...                     // so on for the others
```

Files contain 1 document in every line, prepended with the topics in FastText format (
e.g. `__label__{topic1} __label__{topic2} document`). For more information on this format,
visit the following page: https://fasttext.cc/docs/en/supervised-tutorial.html#getting-and-preparing-the-data.

### RCV1 - RCV2 (news categorization)

1. Request datasets here: https://trec.nist.gov/data/reuters/reuters.html;
    1. Datasets are provided in zipped `.tar.xz` archives.
2. Unzip them in `data/raw/rcv2{lang}/{language}` (e.g. `data/raw/rcv2it/italian` contains the italian articles
   contained in RCV2);
    1. Articles are divided in arbitrary folders (e.g. Italian has 15 folders, `REUTIT1` to `REUTIT15`);
    2. Folders contain `.xml` files, each of which represents a single article;
3. **(Option a).** Generate an individual split using `python/datasets/generation/nc/rcv2.py`.
   This will process the articles from the raw extracted dump. The processed `.xml`
   files are saved as a `.pkl` pickle file to avoid having to re-process them each time a split is created.
   Pickle files are automatically saved in `data/raw/rcv2{lang}/{language}/PICKLE`.

   **(Option b).** You may also generate all splits at the same time, as per step 5.b above.

## Download embeddings

Some methods, like XML-CNN and BiLSTMs, require pretrained embeddings. These may be downloaded
from the following sources, which should then be placed in `data/embeddings/`:

- FastText (multi): https://fasttext.cc/docs/en/crawl-vectors.html
- GloVe (multi): https://www.cs.cmu.edu/~afm/projects/multilingual_embeddings.html
- GloVe (en): https://nlp.stanford.edu/projects/glove

## Usage example

This example describes how to train and test classical methods and
HuggingFace :hugs: Transformers. The procedure is similar for Neural Network based models.
In general, to train and test on different datasets one must:

1. Change the dataset name and adjust hyperparameters in the config file (in folder `configs/`);
2. Set the dataset class in the run script (in folder `experiments/`).

### Training Classical methods

1. Select the dataset class to be used in `experiments/classical.py`;
2. Specify the split to run classical methods on in `configs/classic_run.yml` under the
   `DATASET_NAME` parameter. Since classical methods are usually quite fast, you may set
   this to "_all_" to perform runs on all splits of said dataset;
3. You may adjust other parameters in `classic_run.yml`, including the maximum number of features
   per document representation and parameters for the cross validation procedure.
4. Start running and testing by running `python experiments/classical.py`

### Training Transformers

1. Select the dataset class to be used in `experiments/transformer.py`;
2. Select which pre-trained model to use by setting its index in `configs/transformers_models.yml`;
3. Adjust BS, LR and other parameters in `transformers_run.yml` and set the dataset split to use (`DATASET_NAME`)
    1. Training can be stopped and restarted by using the `RELOAD` and `CHECKPOINT_NAME` parameters;
4. Start training by running `python experiments/transformer.py`.

### Only testing (Transformers)

1. Do training steps 1-3;
2. In `transformers_run.yml` set `TEST = true` and `CHECKPOINT_NAME` to the desired checkpoint file;
    1. Accordingly, `TRAIN` should be set to `false`.
3. Run `python experiments/transformer.py`.

### Running BiLSTMs

This particular architecture was developed in keras rather than torch. In our opinion, keeping a separate
environment for Tensorflow and PyTorch usually avoids a large number of headaches.
The `environment-keras.yml` file is provided to create such an environment. It is based on the conda
distribution of tensorflow, which comes pre-packaged with compiled cuda binaries.
Create and activate this environment, then train with `experiments/bilstm.py` and configure with `configs/keras_rum.yml`.

**Note**: this environment may take a while to install.

## Authors and acknowledgment

- Andrea Gasparetto, Department of Management, Ca' Foscari University, Venice, Italy
- Alessandro Zangari, Department of Management, Ca' Foscari University, Venice, Italy
- Matteo Marcuzzo, Department of Management, Ca' Foscari University, Venice, Italy
- Andrea Albarelli, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy

## Citing this work

If you find this code or the provided data useful in your research, please consider citing:

```
@Article{info13020083,
    AUTHOR = {Gasparetto, Andrea and Marcuzzo, Matteo and Zangari, Alessandro and Albarelli, Andrea},
    TITLE = {A Survey on Text Classification Algorithms: From Text to Predictions},
    JOURNAL = {Information},
    VOLUME = {13},
    YEAR = {2022},
    NUMBER = {2},
    ARTICLE-NUMBER = {83},
    URL = {https://www.mdpi.com/2078-2489/13/2/83},
    ISSN = {2078-2489},
    DOI = {10.3390/info13020083}
}
```

```
@article{10.1371/journal.pone.0270904,
    doi = {10.1371/journal.pone.0270904},
    author = {Gasparetto, Andrea AND Zangari, Alessandro AND Marcuzzo, Matteo AND Albarelli, Andrea},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {A survey on text classification: Practical perspectives on the Italian language},
    year = {2022},
    month = {07},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pone.0270904},
    pages = {1--46},
    number = {7}
}
```

```
@dataset{andrea_gasparetto_2022_7244893,
   author       = {Andrea Gasparetto and
                  Alessandro Zangari and
                  Matteo Marcuzzo and
                  Andrea Albarelli},
   title        = {It/Fr/En-Wiki-100 datasets},
   month        = "jul",
   year         = 2022,
   publisher    = {Zenodo},
   version      = {1.0},
   doi          = {10.5281/zenodo.7244893},
   url          = {https://doi.org/10.5281/zenodo.7244893}
}
```