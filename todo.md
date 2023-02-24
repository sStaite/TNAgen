To do:

    test examples that I have written down


#### Notes:

## Creating the package
From https://packaging.python.org/en/latest/tutorials/packaging-projects/

1) Update pip if needed:
```python3 -m pip install --upgrade build```

2) Then, in the folder with pyproject.toml:
```python3 -m build```

3) Make an account on pypi.prg

4) From the main folder, use: 
```twine upload dist/* ```

5) Then it hopefully should be downloadable with: ```pip install TNA```


## Files

These are all the files that are not for the package, that I just use because they are helpful.

All of examples/ are images that show progress throughout the 7 weeks

TNAgen/generaltest.py  is the main python file that I have been using to test the package.
TNAgen/data/sanity_images is the folder where generaltest.py saves to

noise_removal.ipynb and snr.ipynb are notebooks that I used to test otsu's method and to find the realistic SNRs respectively 


## SNR
At the moment, I am finding the SNR of the glitch before it is cleaned - then saving that requiredSNR/actualSNR. I then clean the spectrogram, convert it to a timeseries then
multiply it by that ratio. 

I worked through SNR that are realistic, but haven't really tested it enough to be certain about it.

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
Would like to explore this but do not have time - Otsu works for glitches which have a larger amout of high intensity pixels. 

If you update the C code, you have to recompile the .so file
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

look at noise_removal.ipynb
OR 
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

## More

Anything more might be at https://tnagen.readthedocs.io/en/latest, otherwise email me at seamus.staite@gmail.com :-)