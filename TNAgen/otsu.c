#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

int square(int i);

int square(int i) {
    return i * i;
}