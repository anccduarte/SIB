# Intelligent Systems for Bioinformatics

## Curricular Unit
Masters in Bioinformatics, Universidade do Minho, 2022-2023.


## Description
A Python package implementing some common machine learning algorithms.
All algorithms are implemented from scratch using numpy and pandas.


## Setup
To get started, fork the repository from GitHub and clone it to your local machine.

Fork the following GitHub repository: https://github.com/anccduarte/SIB

Then, clone the repository to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/SIB.git
```

Open the repository in your favorite IDE and install the dependencies (if missing):
```bash
pip install -r requirements.txt
```
or
```bash
pip install numpy pandas scipy matplotlib
```

## Architecture
The package is organized as follows:
```
si
├── src
│   ├── si
│   │   ├── __init__.py
│   │   ├── data
│   │   │   ├── __init__.py
├── datasets
│   ├── README.md
│   ├── ...
├── scripts
│   ├── README.md
│   ├── ...
├── ... (python package configuration files)
```

A tour to Python packages:
- The _src_ folder contains the source code of the package. It contains an intermediate file 
called _si_, which, in turn, contains all the modules of the package.
- The _datasets_ folder contains the datasets used in the scripts.
- The _scripts_ folder contains the scripts used to test the package and include examples.


## Credits
This package is heavily inspired by [https://github.com/vmspereira/si](https://github.com/vmspereira/si).
