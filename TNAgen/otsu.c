// https://github.com/jpbalarini/otsus/blob/master/algo.cpp
// https://www.ipol.im/pub/art/2016/158/article_lr.pdf

#include <stdio.h>

#define MAX_INTENSITY 1

typedef struct Spectrogram {
    int     a;
    float   array[140][170];
} Spectrogram;

int compute_threshold(Spectrogram *img) {

    /* */


    return 0;
}

void compute_histogram(Spectrogram *img, unsigned *hist){
  // Compute number of pixels
  long int N = 140 * 170;
  int i = 0;

  // Initialize array
  for(i = 0; i <= MAX_INTENSITY; i++) hist[i] = 0;

  // Iterate image
  for (i = 0; i < N; i++) {
    // int value = (int) in[i];
    //hist[value]++;
  }

  printf("Total # of pixels: %ld\n", N);
}