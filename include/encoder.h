// encoder layer and stack

#ifndef ENCODER_H
#define ENCODER_H

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include "tensor.h"

typedef struct {
  // Multi-head self-attention block
  AttentionParams attn_params;
  // First Layer Norm
  LayerNormParams ln1_params;
  // Feedforward Network block
  FeedForwardParams ffn_params;
  // Second Layer Norm
  LayerNormParams ln2_params;
} EncoderParams;

void compute_encoder_layer(const float *X, const EncoderParams *params,
                           float *out, int L, int d_model, int d_ff,
                           int num_heads);

#endif
