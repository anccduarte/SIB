# SIB

## Curricular Unit
SIB - Intelligent Systems for Bioinformatics<br>
Masters in Bioinformatics, University of Minho (2022-2023)


## Description
A _Python_ package implementing some common machine learning algorithms.
All algorithms are implemented from scratch using _numpy_ and _pandas_.


## Setup
To get started, fork the repository from _GitHub_ and clone it to your local machine.

Fork the following _GitHub_ repository: https://github.com/anccduarte/SIB-ML-Portfolio

Then, clone the repository to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/SIB-ML-Portfolio.git
```

Open the repository in your favorite _IDE_ and install the dependencies (if missing):
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

- The _src_ folder contains the source code of the package. It contains an intermediate file 
called _si_, which, in turn, contains all the modules of the package.
- The _datasets_ folder contains the datasets used in the scripts.
- The _scripts_ folder contains the scripts used to test the package and include examples.


## Credits
This package is heavily inspired by [https://github.com/vmspereira/si](https://github.com/vmspereira/si).
