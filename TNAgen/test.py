from ctypes import *
import numpy as np
# env /usr/bin/arch -x86_64 /bin/zsh --login
# gcc -shared -o otsu.so -fPIC otsu.c

data = np.loadtxt("TNAgen/output.txt").tolist()

class Spectrogram(Structure):
    _fields_ = [("a", c_int),
                ("array", (c_float * 170) * 140)]

otsu = CDLL("TNAgen/otsu.so")
otsu.compute.argtypes = [POINTER(Spectrogram)]

t = Spectrogram()

for i in range(140):
    for j in range(170):
        t.array[i][j] = data[i][j]

otsu.compute(byref(t))