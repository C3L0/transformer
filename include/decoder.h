// decoder layer and stack
#ifndef DECODER_H
#define DECODER_H

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"

typedef struct {
  // Masked self attention
  AttentionParams self_attn_params;
  LayerNormParams ln1_params;

  // Cross-Attention
  AttentionParams cross_attn_params;
  LayerNormParams ln2_params;

  // FeedForward Network
  FeedForwardParams ffn_params;
  LayerNormParams ln3_params;
} DecoderLayerParams;

void compute_decoder_layer(const float *dec_input, const float *enc_output,
                           const DecoderLayerParams *params, float *dec_output,
                           int L_dec, int L_enc, int d_model, int d_ff,
                           int num_heads);

#endif
