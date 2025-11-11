// layer normalization
#ifndef LAYERNORM_H
#define LAYERNORM_H

typedef struct {
  float *gamma;
  float *beta;
} LayerNormParams;

void compute_layernorm(const float *input, const LayerNormParams *param,
                       float *output, int L, int d_model);

#endif
