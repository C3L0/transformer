#include "../include/layernorm.h"
#include "../include/utils.h"

#include <math.h>

void compute_layernorm(const float *input, const LayerNormParams *params,
                       float *output, int L, int d_model) {

  for (int i = 0; i < L; i++) {
    //--1-- isolate the current token's feature vercor
    const float *X_i = input + i * d_model;
    float *Y_i = output + i * d_model;
    //--2-- compute the mean and variance
    float mean, var;
    compute_mean_variance(X_i, d_model, &mean, &var);
    //--3-- layer normalization formula
    for (int j = 0; j < d_model; j++) {
      float x_hat = (X_i[j] - mean) / sqrtf(var);
      Y_i[j] = params->gamma[j] * x_hat + params->beta[j];
    }
  }
}
