#include "../include/layernorm.h"
#include "../include/utils.h"

void compute_layernorm(const float *input, const LayerNormParam *params,
                       float *output, int L, int d_model) {

  for (int i = 0; i < L; i++) {
    // isolate the current token's feature vercor
    const float *X_i = input + i * d_model;
    float *Y_i = output + i * d_model;
    // compute the mean and variance
    float mean, var;
    compute_mean_variance(X_i, d_model, &mean, &var);
    // layer normalization formula
    for (int j = 0; j < d_model; j++) {
      float x_hat = (X_i[j] - mean) / sqrtf(var);
      Y_i[j] = params->gamma[j] * x_hat + params->beta[j];
    }
  }
}
