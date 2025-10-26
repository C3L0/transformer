// feed-forward network
#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "config.h"
#include <stddef.h>

typedef struct {
  // Linear 1: W1 (d_model x d_ff)
  float *W1;
  // Bias 1: B1 (d_ff)
  float *B1;
  // Linear 2: W2 (d_ff x d_model)
  float *W2;
  // Bias 2: B2 (d_model)
  float *B2;
} FeedForwardParams;

void compute_feedForward_network(const float *input,
                                 const FeedForwardParams *params, float *output,
                                 int L, int d_model, int d_ff);

#endif
