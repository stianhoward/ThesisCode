# ThesisCode

Code repository joining the functioning components of code used in my thesis.
Exploratory and non-functioning branches are not included as they were never merged back to the main branch and were incompatable as development continued.

Being a thesis with a deadline and not commercial code, the code in this repository is in a 'working' or 'prototype' level of development and is not intended to be scalable or maintainable in this form. Some of the code, most notably for plotting and data preparation, requires reconfiguration for every use.

## Installation
### Python environment

This project used python and C++. 
Python was organized through a conda environment.
With conda installed, the environment can be installed with the following commend from the repository main directory:

`
conda env create -f entironment.yml
`

This will install all packages into an environment named 'thesis'. For a different environment name, edit line 1 of `environment.yml`.

The environment can then be activated by calling 

`
conda activate thesis
`

### C++ environment

The project was built and tested using the gcc version 12.2.0 compiler using C++ 14 on linux.
A binary is included in the relevant build directory, depending on the two models in the parent directories to time them.
The 'long_model.pt' is the model design used for the thesis.

The C++ timing scripts depend on the C++ API for PyTorch, available [here.](https://pytorch.org/get-started/locally/ "here.")

The CMAKE file is then a standard barebones script based off the PyTorch is C++ example.
The code can be compiled using

`
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
`
The file paths to the models are hard coded, and the executable does not require any arguments.


## Repository Structure

This repository is compiles the three development repositories used in developing the thesis.

- Thesis_Data contains the raw data for training. 
- Thesis_Models contains the AICD equation fitting model, machine learning scripts, and the C++ timining scripts
- Thesis_Plots contains some scripts and a jupyter notebook to generate plots and figures used in the thesis.

Configuration of the models, in Thesis_Models, are typically done through global variable definitions immediately following the imports.

Machine learning models are logged to Tensorboard, and can be viewed by running `tensorboard --logdir lightning_logs` in the Thesis_Models/ml/models directory.


## Primary packages/libraries used

Python:
- Scipy
- Numpy
- Pandas
- matplotlib
- Pytorch
- PyTorch-lightning

C++
- PyTorch C++ API



