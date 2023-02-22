// https://github.com/jpbalarini/otsus/blob/master/algo.cpp
// https://www.ipol.im/pub/art/2016/158/article_lr.pdf

#include <stdio.h>
#include <math.h>

#define HISTOGRAM_SIZE 256

typedef struct Spectrogram {
    int     a;
    float   array[140][170];
} Spectrogram;

void compute_histogram(Spectrogram *img, int hist[]){
  // Compute number of pixels
    for (int i = 0; i < 140; i++) {
        for (int j = 0; j < 170; j++) {
            int pixel_value = (int)(img->array[i][j] * HISTOGRAM_SIZE);
            hist[pixel_value]++;
        }
    }
}

float compute_sum_of_pixels(int hist[]) {
    // Compute sum of pixel values
    int sum = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        sum += i * hist[i];
    }
    return sum;
}


int compute_threshold(Spectrogram *img) {

    int hist[HISTOGRAM_SIZE] = {0};
    compute_histogram(img, hist);

    int total_pixels = 140*170;
    
    float sum = compute_sum_of_pixels(hist);
    float sum_between = 0;

    // Compute maximum variance and threshold value
    float max_variance = -1;
    int best_threshold = 0;
    int background_pixels = 0; // q1
    int foreground_pixels = 0; // q2

    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        background_pixels += hist[i];
        foreground_pixels = total_pixels - background_pixels;
        
        if (foreground_pixels == 0) {
            break;
        }
        
        if (background_pixels == 0) {
            continue;
        }
        
        sum_between += (float) (i * ((int)hist[i]));
        float mean1 = sum_between / background_pixels;
        float mean2 = (sum - sum_between) / foreground_pixels;
        
        float between_variance = (float) background_pixels * (float) foreground_pixels * pow((mean1 - mean2), 2);

        if (between_variance > max_variance) {
            max_variance = between_variance;
            best_threshold = i;
        }
    }
    
    return best_threshold;
}

void clean_image(Spectrogram *img) {
    float threshold = (float) compute_threshold(img) / (float)HISTOGRAM_SIZE;
    for (int i = 0; i < 140; i++) {
        for (int j = 0; j < 170; j++) {
            if (img->array[i][j] < threshold) {
                img->array[i][j] = 0;
            }
        }
    }
}

