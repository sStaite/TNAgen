

# TNAgen
## Introduction
TNAgen is a python package that allows users to generate transient noise artifacts in gravitational-wave detector data using a GAN model. The package is based on the paper "Generating Transient Noise Artefacts in Gravitational-Wave Detector Data with Generative Adversarial Networks" by Powell et. al. It uses the torchGAN library to implement the GAN model and provides a user-friendly interface for generating noise artifacts for specific glitches.

## Table of contents
* [Installation](#Installation)
* [Documentation](#Documentation)
* [Contributors](#Contributors)

## Installation

To install environment

```console
$ conda env create --name myenv --file=requirements.yml
$ conda activate myenv
```

To install:
```console
$ pip install TNAgen
```

```console
$ pip install -i https://pypi.anaconda.org/sstaite/simple tnagen
```

From Source:
```console
$ git clone https://github.com/sStaite/TNAgen
$ cd TNAgen
$ python setup.py install
```

## Documentation
Documentation can be found at https://tnagen.readthedocs.io/


## Contributors
Seamus Staite, Jade Powell

