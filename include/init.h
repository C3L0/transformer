// parameter initialization
#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"

void init_attention_params(AttentionParams *params, int d_model, int num_heads,
                           int random_init);

void free_attention_params(AttentionParams *params);

void init_layernorm_params(LayerNormParams *params, int d_model);

void free_layernorm_params(LayerNormParams *params);

void init_feedforward_params();

void free_feedforward_params(FeedForwardParams *params);
