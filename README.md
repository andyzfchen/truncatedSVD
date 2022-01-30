# ML Reproducibility Challenge 2021

This repository hosts documents and code for reproducing the algorithm for updating the truncated SVD of evolving matrices outlined by Vassilis Kalantzis, Georgios Kollias, Shashanka Ubaru, Athanasios N. Nikolakopoulos, Lior Horesh, and Kenneth L. Clarkson in their paper [Projection techniques to update the truncated SVD of evolving matrices](http://proceedings.mlr.press/v139/kalantzis21a/kalantzis21a.pdf) published in the 38th International Conference on Machine Learning 2021.
We (Andy Chen, Shion Matsumoto, and Rohan Sinha Varma) present this repository as part of a submission to the [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021) along with our [report]().

## How to Run Experiments

The following sections provide instructions on the installation in order to run the experiments as well as the datasets used in the study.

### Installation

The code was written entirely in Python using standard packages.

### Datasets

The datasets have been converted into a convenient format and can be accessed [here](https://drive.google.com/drive/folders/1tHrUILY_NBKDPmNYOaEpWnc9-1US9DEB). Note that a few of the datasets had to be converted from a sparse into dense format.

### Experiment Specifications

The experimental parameters are specified in a JSON file as follows:

```json
{
  "tests": [
    {
      "method": "zha-simon",
      "datasets": ["CISI", "MED", "CRAN"],
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
}
```

Below is a table listing parameters and their descriptions. Note that some methods may require additional parameters. Please see our `example.json` file for a complete example for all update methods.

| Parameter    | Description                                  | Example                        |
| ------------ | -------------------------------------------- | ------------------------------ |
| dataset_info | Name and location of dataset                 | "CRAN": "../datasets/CRAN.npy" |
| method       | Update method                                | ["CISI", "MED", "CRAN"]        |
| m_percent    | Percent of                                   | 0.1                            |
| n_batches    | Number of update batches                     | 10                             |
| phis_to_plot | Batch numbers to plot                        | [1, 5, 10]                     |
| k_dims       | Rank of updates                              | [25, 50, 100]                  |
| make_plots   | Option to plot update results                | true                           |
| r_values     | Number of (BCG only)                         | [10, 20, 30, 40, 50]           |
| lam_coeff    | Coefficient (BCG only)                       | 1.01                           |
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
