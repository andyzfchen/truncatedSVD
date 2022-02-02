# ML Reproducibility Challenge 2021

This repository hosts documents and code for reproducing the algorithm for updating the truncated SVD of evolving matrices outlined by Vassilis Kalantzis, Georgios Kollias, Shashanka Ubaru, Athanasios N. Nikolakopoulos, Lior Horesh, and Kenneth L. Clarkson in their paper [Projection techniques to update the truncated SVD of evolving matrices](http://proceedings.mlr.press/v139/kalantzis21a/kalantzis21a.pdf) published in the 38th International Conference on Machine Learning 2021.
We (Andy Chen, Shion Matsumoto, and Rohan Sinha Varma) present this repository as part of a submission to the [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021) along with our [report]().

## How to Run Experiments

The following sections provide instructions on the installation in order to run the experiments as well as the datasets used in the study.

### Conda Environment Setup

The necessary dependencies are listed in the [`environment.yml`](environment.yml) file and can be used to set up a new [Anaconda](https://www.anaconda.com/) environment for running the experiments in this repository.

```bash
conda env create -n <env-name> -f environment.yml
conda activate <env-name>
```

### Datasets

The datasets have been converted into a convenient format and can be accessed [here](https://drive.google.com/drive/folders/1tHrUILY_NBKDPmNYOaEpWnc9-1US9DEB). Note that a few of the datasets had to be converted from a sparse into dense format.

### Experiment Specifications

The experimental parameters are specified in a JSON file as follows:

```json
{
  "tests": [
    {
      "dataset": "CRAN",
      "method": [
        "zha-simon",
        "bcg",
        "fd"
      ],
      "m_percent": 0.1,
      "n_batches": [10],
      "phis_to_plot": [1, 5, 10],
      "k_dims": [50],
      "make_plots": true
    }
  ],
  "dataset_info": {
    "CISI": "../datasets/CISI.npy",
    "CRAN": "../datasets/CRAN.npy",
    "MED": "../datasets/MED.npy"
  }
  "method_label": {
    "bcg": "$Z = [U_k, X_{\\lambda,r}; 0, I_s]$",
    "zha-simon": "$Z = [U_k, 0; 0, I_s]$",
    "frequent-direction": "FrequentDirections",
  }
}
```

Below are tables listing parameters and their descriptions. Please see our JSON files in the experiments directory for complete examples.

| Parameter    | Description                                  | Example                        |
| ------------ | -------------------------------------------- | ------------------------------ |
| tests        | List of json objects describing tests        | See table below                |
| dataset_info | Name and location of datasets used in tests  | "CRAN": "../datasets/CRAN.npy" |
| method label | Labels used in plots for each method         | "zha-simon": "Zha Simon"       |

The ```tests``` parameter provides a list of json objects specifying all the tests to be run. Below we detail what these JSON objects must contain. Note if BCG is being run on any dataset, the BCG only parameters must be included

| Parameter    | Description                                  | Example                        |
| ------------ | -------------------------------------------- | ------------------------------ |
| dataset      | Name of dataset to run on                    | "CRAN"                         |
| method       | List of update methods to run                | ["zha-simon", "bcg", "fd"]     |
| m_percent    | Percent of data used as initial matrix       | 0.1                            |
| n_batches    | Number of update batches                     | 10                             |
| phis_to_plot | Batch numbers to plot                        | [1, 5, 10]                     |
| k_dims       | Rank of updates                              | [25, 50, 100]                  |
| make_plots   | Option to plot update results                | true                           |
| r_values     | Number of oversamples(BCG only)              | [10, 20, 30, 40, 50]           |
| lam_coeff    | Lambda Coefficient (BCG only)                | 1.01                           |
| num_runs     | Number of runs for BCG experiment (BCG only) | 1                              |

To run the experiment, all you have to do is call `run_tests.py` and specify the path to the JSON file and the directory to contain the cache folder:

```shell
python run_tests.py <tests.json> <cache_directory>
```

**Note:** The cache folder becomes very large (~5 GB), so please check that your system has sufficient space before running the experiments.

## Results

Below are plots for various error metrics based on the experiments we conducted for our report. To reproduce our results ...

## Team

Andy Chen, Shion Matsumoto, Rohan Varma

## Acknowledgments

We would like to acknowledge Professor Laura Balzano for introducing us to this challenge and also for advising us on this project. We would also like to thank Professor Vassilis Kalantzis for providing us with the code.
