# NPSN: Nuclear Power Surrogate Network

NPSN is a package that allows easy training and optimization of [ANN](https://github.com/tensorflow/tensorflow)/[GBR](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)/[GPR](https://github.com/GPflow/GPflow)/[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR) algorithms to provide multidimensional regression of a nuclear reactor's power distribution based on control blade position(s).
The package is written to abstract the process of importing/pre-processing data, optimizing hyperparameters, and providing performance metrics.
The aim of this project is to facilitate development of empirical models that are needed in autonomous reactor control systems.
Format for training data is detailed at the top of the [data generation](npsn/dg.py) script.

## Example
```python
import npsn

# Define dataset directory
data_dir = '~/some/data_location'
# Define project name (for output file label)
proj_nm = 'npsn_surrogate'

# Define algorithm type to be used
algo_nm = "ANN"
# Define number of control devices
n_x = 4
# Define nodalization of power distribution
n_y = (15, 20)  # (axial_nodes, fuel_locations)

# Train without optimization
npsn.train(proj_nm, algo_nm, data_dir, n_x, n_y)
# OR with optimization
npsn.train(proj_nm, algo_nm, data_dir, n_x, n_y, max_evals=100)
# Post-process to quantify error (CSV file output)
npsn.post(proj_nm)
```

The model will be saved in the current working directory (_/cwd_).
Error metrics will be output to the _/cwd/csv_ directory and consist of mean and standard deviation of MAP error against test and training data.
If optimization studies are conducted, the data on each permutation will be output to the _/cwd/mat_ directory and consist of a .mat file that can be loaded into MATLAB or with `scipy.io.loadmat`.

## Installation

To install with pip:
```
pip install npsn
```
The dependency requirements will be satisfied by pip. A full list of the environment used is in [requirements](requirements.txt). 
The package was developed on Ubuntu 18.04, but is written to also work on Mac and Windows OS.

## Paper using NPSN

Initially, NPSN was used to create an [ANN model for the MIT reactor](https://arxiv.org/abs/2007.05435):
![](assets/mitr-surrogate.gif)

NPSN was also used to create [ANN/GBR/GPR/SVR models for the MIT reactor and MIT graphite pile](https://arxiv.org/abs/2105.14645).

## Cite 

If you use NPSN in your work, please cite as:
> A. J. Dave, J. Wilson, K. Sun, “Deep Surrogate Models for Multi-dimensional Regression of Reactor Power,” ANS Winter Conference 2020 (preprint: https://arxiv.org/abs/2007.05435) 

## Contact
If you have any questions, comments, or suggestions feel free to [email](mailto:akshayjd@mit.edu) me!
