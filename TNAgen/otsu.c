
#include <stdio.h>

typedef struct Spectrogram {
    int     a;
    float   array[140][170];
} Spectrogram;

int compute(Spectrogram *img) {

    for (int i = 0; i < 140; i++) {
        for (int j = 0; j < 170; j++) {
            printf("%f ", img->array[i][j]);
        }
        printf("\n");
    }
    return 0;
}