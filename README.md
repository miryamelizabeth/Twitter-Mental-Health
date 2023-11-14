# Twitter-Mental-Health
This repository contains the materials related to the article [Extracting Mental Health Indicators from English and Spanish Social Media: A Machine Learning Approach](https://ieeexplore.ieee.org/document/10315126).



## Requirements

* [Anaconda](https://www.anaconda.com/products/distribution)
* [Python 3](https://www.python.org/downloads/) = 3.8.5
* [Scikit-learn](https://scikit-learn.org/) = 1.02
* [Pandas](https://pandas.pydata.org/) = 1.4.2
* [Numpy](https://numpy.org/) = 1.21.5
* [Dask](https://www.dask.org/)
* [Pysentimiento](https://github.com/pysentimiento/pysentimiento)
* [Spacy](https://spacy.io/) = 3.3.0
* [Tweepy](https://www.tweepy.org/)
* [Pycm](https://www.pycm.io/) = 3.0

[Here](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/env_anaconda.txt) is the anancoda environment with these libraries to do everything that follows.

## Dataset

The Twitter dataset for mental disorder detection is available for download through the [IEEEDataPort](https://ieee-dataport.org/documents/twitter-dataset-mental-disorders-detection) platform.

## Models

### 0) Preprocessing
[Code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/preprocessing_text.py) for preprocessing the dataset.

### 1) Dataset Format
The experiments were run using 5-fold cross-validation. In each partition the distribution of classes is respected. The ``Partitions`` folder has 3 folders inside:

* 5FCV_Txt_ENG
* 5FCV_Pos_ENG
* 5FCV_LfLiwc_ENG

Each folder looks like this:
<p align="center">
    <img src="https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/images/Imagen1.png" width="300">
</p>

5 training files and 5 testing files. The nomenclature of each of them is shown in the image.

Each instance of the dataset is the complete timeline of the user with his/her corresponding class. In the case of Txt and Pos, each file consists of 2 columns:
* ``tweets_user``: Corresponds to all tweets of each user concatenated in a single row. In Txt the processed file column ``clean_tweet_tweet_nostop_lemma`` is used and in Pos ``clean_tweet_nostop_lemma_postags`` is used.
* ``class``: Refers to class label (ADHD, ANXIETY, ASD, BIPOLAR, CONTROL, DEPRESSION, EATING, OCD, PTSD, SQUIZOPHRENIA)

| **Partition** | **Format dataset (columns)** | **Attributes using this folder** | **Description** | **Column used to generate the dataset<br>(from the preprocessed file)** |
|---|---|---|---|---|
| 5FCV_Txt_ENG | `tweets_user`,`class`<br>[example1](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/examples/example1.csv) | n-grams<br>q-grams<br>Topic modeling<br>Embeddings | User's lemmatized words, either as unigrams, bigrams, q-grams, etc. | clean_tweet_nostop_lemma |
| 5FCV_Pos_ENG | `tweets_user`,`class`<br>[example2](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/examples/example2.csv) | Pos n-grams | Lemmatized words and their POS (part-of-speech) tag. For example: we_PRON run_VERB, etc. | clean_tweet_nostop_lemma_postags |
| 5FCV_LfLiwc_ENG | `Dic`,`I`,`Ppron`,...,`class`<br>[example3](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/examples/example3.csv) | LIWC | Numerical attributes extracted from the LIWC2015 tool. | clean_tweet_lemma |

### 2) Machine learning models
The paths to the dataset and results directories must be changed, as well as the language (English/Spanish).
* [Code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/models_binary_ML.py) for Binary models
* [Code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/models_multiclass_ML.py) for Multiclass models

The parameters of `max_features`,`n_topics` and those of the classifiers are already there.

### 3) Neural networks models
The embeddings files should be in an `Embeddings_ENG\Embeddings_ESP` (depending on the language) folder:

* `fasttext-crawl-300d.vec` → https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
* `glove-crawl-300d.txt` → https://nlp.stanford.edu/data/glove.42B.300d.zip

All the parameters are already there, but the paths to the dataset and results directories must be changed, as well as the language (English/Spanish).
* [Code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/models_binary_DL.py) for Binary models
* [Code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/models_multiclass_DL.py) for Multiclass models

### 3) Results
Once the experiments are completed, the results folder looks like this:
<p align="center">
    <img src="https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/images/Imagen2.png" width="800">
</p>

Each folder corresponds to the attribute to be executed. Three types of files are being stored in each fold: the confusion matrix, the normalized confusion matrix and the result of various metrics (class breakdown and Macro-Average).

As long as the nomenclature is respected, folders can be created manually, however, to save time this [code](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/create_folders.py) is used.

[Here](https://github.com/miryamelizabeth/Twitter-Mental-Health/blob/main/save_and_format_results.py) is the code to process the results. *The directories and the name of the final file, located at the bottom, must be changed.*

In the first part, a CSV file is obtained with all the results grouped by mean and standard deviation of each fold. This contains all the classifiers and metrics. In the second part, the code changes the name of the classifiers, metrics, features... to format the results in excel and make them look nice.



## References

For more information, please read: M. E. Villa-Pérez, L. A. Trejo, M. B. Moin and E. Stroulia, "Extracting Mental Health Indicators from English and Spanish Social Media: A Machine Learning Approach," in IEEE Access, doi: 10.1109/ACCESS.2023.3332289.

If you use this data, please cite:
```
@MISC {dataset-twitter-mental-health,
    author    = "Villa-Pérez, Miryam Elizabeth",
    title     = "Twitter Dataset for Mental Disorders Detection",
    month     = "feb",
    year      = "2023",
    doi       = "10.21227/6pxp-4t91",
    url       = "https://dx.doi.org/10.21227/6pxp-4t91",
    publisher = "IEEE Dataport"
}

```
