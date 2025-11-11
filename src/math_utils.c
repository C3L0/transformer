#include "../include/math_utils.h"
#include "../include/utils.h"

#include <math.h>

void softmax_rows(const float *scores, float *weights, int L) {
  for (int i = 0; i < L; i++) {
    float max_val = -INFINITY, sum = 0.0f;
    for (int j = 0; j < L; j++) {
      float val = scores[i * L + j];
      if (val > max_val)
        max_val = val;
    }
    for (int j = 0; j < L; j++) {
      float e = expf(scores[i * L + j] - max_val);
      weights[i * L + j] = e;
      sum += e;
    }
    for (int j = 0; j < L; j++)
      weights[i * L + j] /= sum;
  }
}

void compute_mean_variance(const float *M, int size, float *mean_out,
                           float *var_out) {
  if (size == 0) {
    *mean_out = 0.0f;
    *var_out = 0.0f;
    return;
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += M[i];
  }
  float mean = sum / (float)size;

  float sum_sq_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = M[i] - mean;
    sum_sq_diff += diff * diff;
  }

  float variance = sum_sq_diff / (float)size;

  *mean_out = mean;
  *var_out = variance;
  if (variance == 0) {
    // Add a epsilon to the variance to prevent division by zero in layernorm.c
    *var_out = variance + EPSILON;
  }
}

void apply_gelu(float *M, int rows, int cols) {
  int size = rows * cols;

  for (int i = 0; i < size; i++)
    M[i] = 0.5f * M[i] *
           (1.0f + tanhf(SQRT_2_OVER_PI * (M[i] + GELU_A * powf(M[i], 3))));
}
