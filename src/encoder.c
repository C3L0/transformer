#include "../include/encoder.h"

#include "../include/attention.h"
#include "../include/feedforward.h"
#include "../include/init.h"
#include "../include/layernorm.h"
#include "../include/tensor.h"

#include <stdio.h>
#include <stdlib.h>

void compute_encoder_layer(const float *X, const EncoderParams *params,
                           float *out, int L, int d_model, int d_ff,
                           int num_heads) {

  // --- 1. Allocate intermediate buffers ---
  float *H1 = (float *)calloc(L * d_model, sizeof(float));
  float *H1_res = (float *)calloc(L * d_model, sizeof(float));
  float *H2 = (float *)calloc(L * d_model, sizeof(float));
  float *H2_res = (float *)calloc(L * d_model, sizeof(float));

  if (!H1 || !H1_res || !H2 || !H2_res) {
    fprintf(stderr, "Memory allocation failed in encoder layer.\n");
    exit(1);
  }

  // --- 2 Multi-head attention ---
  compute_multihead_attention(X, &params->attn_params, H1, L, d_model,
                              num_heads);

  // --- 3 Add & Norm (first LayerNorm) ---
  matsum(X, H1, H1_res, L * d_model);
  compute_layernorm(H1_res, &params->ln1_params, H1_res, L, d_model);

  // --- 4 Feedforward network ---
  compute_feedforward_network(H1_res, &params->ffn_params, H2, L, d_model,
                              d_ff);

  // --- 5 Add & Norm (second LayerNorm) ---
  matsum(H1_res, H2, H2_res, L * d_model);
  compute_layernorm(H2_res, &params->ln2_params, out, L, d_model);

  // --- 6. Free temporary buffers ---
  free(H1);
  free(H1_res);
  free(H2);
  free(H2_res);
}
