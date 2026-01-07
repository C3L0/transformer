#include "../include/decoder.h"
#include "../include/attention.h"
#include "../include/feedforward.h"
#include "../include/layernorm.h"
#include "../include/tensor.h"

#include <stdio.h>
#include <stdlib.h>

void compute_decode_layer(const float *dec_input, const float *enc_output,
                          const DecoderLayerParams *params, float *dec_output,
                          int L_dec, int L_enc, int d_model, int d_ff,
                          int num_heads) {

  // helper buffer
  size_t buf_size = L_dec * d_model * sizeof(float);
  float *A1 = (float *)calloc(L_dec * sizeof(float)); // self attn out
  float *Y1 = (float *)calloc(L_dec * sizeof(float)); // post ln1
  float *A2 = (float *)calloc(L_dec * sizeof(float)); // cross attn out
  float *Y2 = (float *)calloc(L_dec * sizeof(float)); // post ln2
  float *FF = (float *)calloc(L_dec * sizeof(float)); // ffn output

  if (!A1 || !Y1 || !A2 || !Y2 || !FF)
    fprintf(stderr, "Alloc failed in decoder\n", exit(1));

  // Mask Self_attention
  compute_multihead_attention(dec_input, &params->self_attn_params, A1, L_dec,
                              d_model, num_heads);

  // add & norm
  matsum(dec_input, A1, A1, L_dec * d_model);
  compute_layernorm(A2, &params->ln1_params, Y1, L_dec, d_model);

  // Cross Attention
  compute_cross_attention(Y1, enc_output, &params->cross_attn_params, A2, L_dec,
                          L_enc, d_model, num_heads);

  // add & norm
  matsum(Y1, A2, A2, L_dec * d_model);
  compute_layernorm(A2, &params->ln2_params, Y2, L_dec, d_model);

  // Feed-Forward
  compute_feedforward_network(Y2, &params->ffn_params, FF, L_dec, d_model,
                              d_ff);
  // add & norm
  matsum(Y2, FF, FF, L_dec * d_model);
  compute_layernorm(FF, &params->ln3_params, dec_output, L_dec, d_model);

  // cleanup
  free(A1);
  free(Y1);
  free(A2);
  free(Y2);
  free(FF);
}
