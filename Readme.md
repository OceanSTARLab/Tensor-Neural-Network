# Tensor Neural Network

This repository contains the code for the paper "Striking The Right Balance: Three-dimensional Ocean Sound Speed Field Reconstruction Using Tensor Neural Networks".

## Description

The code for model training and evaluation can be found in `main.py`. The `data.mat` file contains the demeaned 3D SSF data, and `data_mean.mat` provides the data mean. Additionally, `util.py` contains functions for tensor operations.

## Usage

To use the code, open a command line and run:

```cmd
python main.py
```

You can change the hyper-parameters using the command line. For example:

```cmd
python main.py --p 0.4 --sigma 0.3 --lamb 5 --runs 12000 --verbose True
```

Please refer to the paper and `main.py` for more details on the hyper-parameters and their effects.
