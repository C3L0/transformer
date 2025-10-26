#include "../include/feedforward.h"
#include "../include/utils.h"

#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

void compute_feedforward_network(const float *input,
                                 const FeedForwardParams *params, float *output,
                                 int L, int d_model, int d_ff) {

  //--1-- Linear Layer 1: H1 = X * W1 + B1
  float *H1_pre_act = calloc((size_t)L * d_ff, sizeof(float));
  if (!H1_pre_act)
    return;

#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_ff, d_model, 1.0f,
              input, d_model, params->W1, d_ff, 0.0f, H1_pre_act, d_ff);

#else
  matmul_blocked(input, params->W1, H1_pre_act, L, d_ff, d_model);

#endif
  matrix_add_vector_bias(H1_pre_act, params->B1, L, d_ff);

  //--2-- Activation: H1 = GeLu(H1_pre_act)
  apply_gelu(H1_pre_act, L, d_ff);

  //--3-- Linear Layer 2: Output = H1 * W2 + B2
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_model, d_ff, 1.0f,
              H1_pre_act, d_ff, params->W2, d_model, 0.0f, output, d_model);

#else
  matmul_blocked(H1_pre_act, params->W2, output, L, d_model, d_ff);

#endif
  matrix_add_vector_bias(output, params->B2, L, d_model);

  free(H1_pre_act);
}
