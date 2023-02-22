To do:

    look at paper jade sent - generate the whistle glitch from a mathematical formula

    whats going on with SNR 


Notes:

## Git LFS
Git LFS will be a good option for the models - 19 models that take up about 1.06gb of space. Unfortunately I have used over 1gb so github banned me for a month from using it.

## Installation
A Conda environment must be used - this is because python-framel (the python package that is used to write data to .gwf files) can only be managed by Conda. 
Annoyingly, torchGAN can only be installed via pip. This means the requirements are stored in a requirements.yml file. 

Installed with conda:
pytorch
matplotlib
numpy
pandas
librosa, libvorbis (this was need for some reason? even though its a dependency of librosa)
scipy
gwpy
python-framel
h5py

Installed with pip:
torchGAN

## Otsu's method
Otsu's method works - definitely could be worth exploring in _clean_spectrogram() function, to find the threshold -> could replace 'spectrogram[spectrogram < threshold] = 0'
Would like to explore this but do not have time.

If you update the C code, you must recompile the .so file
Since I have a M1 chip in my macbook, I needed to change my terminal architechure (I don't really know what this means!)
To do this, I used
```console
$ env /usr/bin/arch -x86_64 /bin/zsh --login
$ arch
```
This has to return a value of 'x86_64' or 'i386', if it is 'arm64' the next line (to compile) won't work
```console
$ gcc -shared -o otsu.so -fPIC otsu.c
```

How to use:
```python
from ctypes import * # This is a default package for importing C code into python

# Special structure that C can recognise
class Spectrogram(Structure):
    _fields_ = [("a", c_int),
                ("array", (c_float * 170) * 140)]

# Import the file
otsu = CDLL("otsu.so")

# Specify what the object that going INTO the C code is.
otsu.clean_image.argtypes = [POINTER(Spectrogram)]
otsu.compute_threshold.argtypes = [POINTER(Spectrogram)]

t = Spectrogram()
d = data.astype(np.float32) # Need to convert our data to float32 instead of float64
memmove(byref(t.array), d.ctypes.data, d.nbytes) # Copy our data into the Spectrogram structure, by memory allocation (this is much faster than two for loops)

# To clean the image
otsu.clean_image(byref(t)) # Give the reference to the structure to the 'clean_image' function - it mutates t.array, so we don't need a return value.
cleaned = np.array(t.array)
plt.imshow(cleaned)
plt.show()

# To find the threshold
threshold = otsu.compute_threshold(byref(t)) # This is a number from 0 to 255, so scale it to [0, 1]
threshold /= 255
```
