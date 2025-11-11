// parameter initialization
#ifndef INIT_H
#define INIT_H

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"

void fill_random(float *M, int N);

void init_attention_params(AttentionParams *params, int d_model, int num_heads,
                           int random_init);

void free_attention_params(AttentionParams *params);

void init_layernorm_params(LayerNormParams *params, int d_model);

void free_layernorm_params(LayerNormParams *params);

void init_feedforward_params(FeedForwardParams *params, int d_model, int d_ff);

void free_feedforward_params(FeedForwardParams *params);

#endif
