# SETI Breakthrough Listen - E.T. Signal Search

This repository aims to analyze the dataset from the Kaggle challenge ["SETI Breakthrough Listen - E.T. Signal Search"](https://www.kaggle.com/c/seti-breakthrough-listen/overview) and compare state-of-the-art [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) CNN's under the aspect of data augmentation to answer the question whether data augmentation helps in reaching a better score or not.

## Table of contents

- [SETI Breakthrough Listen - E.T. Signal Search](#seti-breakthrough-listen---et-signal-search)
  - [Table of contents](#table-of-contents)
  - [About the challenge](#about-the-challenge)
  - [About this repository](#about-this-repository)
  - [How to execute the code](#how-to-execute-the-code)
    - [Dealing with CUDA out of memory error](#dealing-with-cuda-out-of-memory-error)


## About the challenge

["SETI Breakthrough Listen - E.T. Signal Search"](https://www.kaggle.com/c/seti-breakthrough-listen/overview) is a challenged hosted on kaggle which deals with classifying data samples as extraterrestrial or not. Those data samples so-called *cadence snippets*. This is a set of six observations of the order ABACAD in which the primary target A is alternatingly observed along with other targets B, C and D. Those observations are spectrograms which consist of measurements of signal intensity as a function of frequency and time (see [here](https://www.kaggle.com/c/seti-breakthrough-listen/overview/data-information)).

To find extraterrestrial signals, one can isolate them by analyzing the cadence snippet. Such an extraterrestrial signal can only appear in the observations from the primary target.

The aim of the challenge is to build a classification model which helps to identify the cadences with anomalous signals. The metric used here is the area under the ROC curve.

## About this repository
The codebase in this repository consists only of a Jupyter Notebook in which the whole process of loading the data, defining and training the networks and visualizing the data is realized.

For this process the EfficientNet models B1, B3 and B5 are used as the networks. Those are trained and validated solely on the train data as it would be necessary to submit the results to check the score based on the test results. This has not be done so the results cannot be verified to the test data.

## How to execute the code
First, there are dependencies which are linked to the challenge and the rules applied to it, so the dataset and the annotations cannot be provided in the repository. One must create an account at [Kaggle](https://www.kaggle.com/) and join the competition to download the dataset. There are also pretrained networks used in this notebook which have been provided from Henrique Mendonça [here](https://www.kaggle.com/hmendonca/efficientnet-pytorch). Those are of course not mandatory: the code can also be executed without using pretrained weights.

After resolving the dependencies, one can create an environment with all the packages needed with conda. This can be done by using this command:
```
conda env create --name [name] -f environment.yml
```

This process can take around 10 to 20 minutes. After the environment has been created, one can activate it and start the notebook with jupyter.
```
> conda activate [name]

> ...

> jupyter-notebook modelComparison.ipynb
```

### Dealing with CUDA out of memory error
When trying to execute this notebook, one will often face the problem that CUDA ran out of memory when trying to allocate. The reason for this is the complexity from the used models, especially EfficientNet-B5. It is not unusual that it can take 15 GiB of VRAM when using a batch size of 16 for it. As only very few GPU's can offer that much of memory, one can reduce the memory usage from PyTorch with two methods.

First, you can reduce the batch size in the notebook.

Second, you can set max_split_size_mb defining it in the `PYTORCH_CUDA_ALLOC_CONF` environment variable. To set this for example to 64 MB, one can define:

```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

More information to this can be found [here](https://pytorch.org/docs/stable/notes/cuda.html#memory-management).

Both solutions lead (depending on the values used) to a substantial performance cost which means that the training process takes much more time.
