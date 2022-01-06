# ML Reproducibility Challenge 2021

This repository holds the relevant documents and code for reproducing the Truncated SVD algorithm for evolving matrices outlined by Vassilis Kalantzis, Georgios Kollias, Shashanka Ubaru, Athanasios N. Nikolakopoulos, Lior Horesh, and Kenneth L. Clarkson in their paper ``Projection techniques to update the truncated SVD of evolving matrices'' published in the 2021 Thirty-eighth International Conference on Machine Learning.
We (Andy Chen, Shion Matsumoto, and Rohan Sinha Varma) present this repository as an entry into the 2021 Machine Learning Reproducibility Challenge.

## File Organization

The Python modules and datasets have been organized in such a way that they can be accessed easily by one another.

## How to Run Experiments

The following sections provide instructions on the installation in order to run the experiments as well as the datasets used in the study.

### Installation

The code was written entirely in Python using standard packages.

### Datasets

The datasets have been converted into a convenient format and can be accessed [here]().

Their original sources can be found [here](). Note that a few of the datasets had to be converted from a sparse into dense format.

### Experiment Specifications

The experimental parameters are specified in a `specs.json` file with the following parameters:

```json
{
    "datasets": ["CRAN", "CISI", "MED"],
    "overwrite": "y",
}

| Parameter | Description                                | Value |
| --------- | ------------------------------------------ | ----- |
| Datasets  | Name of datasets to be used in experiments | ----  |

## Results



## Examples

Below is a set of commands to run one of our experiments:

```
# navigate to experiment directory
cd [...]/truncatedSVD

# run experiments
python truncatedSVD.py

# create error plots
python make_plots.py
```

## Team


## ML Reproducibility Submission

This study was submitted to the [ML Reproducibility 2021 Challenge](https://paperswithcode.com/rc2021). The full report can be found [here]().
