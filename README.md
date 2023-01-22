

# TNAgen
## Introduction
TNAgen is a python package that allows users to generate transient noise artifacts in gravitational-wave detector data using a GAN model. The package is based on the paper "Generating Transient Noise Artefacts in Gravitational-Wave Detector Data with Generative Adversarial Networks" by Powell et. al. It uses the torchGAN library to implement the GAN model and provides a user-friendly interface for generating noise artifacts for specific glitches.

## Table of contents
* [Installation](#Installation)
* [Documentation](#Documentation)
* [Examples](#Examples)
* [Contributors](#Contributors)

## Installation

Using pip: 
```console
$ pip install NOT_READY_YET
```

Using conda:
```console
$ conda install NOT_READY_YET
```

From Source:
```console
$ git clone NOT_READY_YET
$ cd TNAgen
$ python setup.py install
```

## Documentation
Documentation can be found at https://tnagen.readthedocs.io/en/stable/

## Examples
First, we need to import the Generator module and instatiate a Generator object.
```python
from TNAgen import Generator
generator = Generator()
```
Generating and saving 50 Koi Fish glitches as images of spectrograms:
```python
generator.generate("Koi_Fish", 50)
generator.save_as_png("path/to/file", clear_queue=True)
```
Generating and saving 2 of each glitch and saving as time-series in a hdf5 file:
```python
generator.generate_all(2)
generator.save_as_hdf5("path/to/file", "name", clear_queue=True)
```
Generating raw data (numpy arrays) for 50 Blip glitches.
```python
numpy_array, label_list = generator.generate("Blip", 50)
generator.clear_queue()
```
Generating 10 Blip glitches, 10 Koi Fish glitches and saving to one hdf5 file, then generating 50 Chirp glitches and saving in another hdf5 file.
```python
generator.generate("Blip", 10)
generator.generate("Koi Fish", 10)
generator.save_as_hdf5("path/to/file", "file1", clear_queue=True)
generator.generate("Chirp", 50)
generator.save_as_hdf5("path/to/file", "file2", clear_queue=True)
```

## Contributors
5

