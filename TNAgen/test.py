import ctypes

otsu = ctypes.CDLL("/Users/seamus/Desktop/OzGrav/TNAgen/TNAgen/otsu.so")
print(otsu.square(2))

# this only works with a homebrew environment! look into this