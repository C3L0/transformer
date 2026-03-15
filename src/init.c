#include "../include/init.h"
#include "../include/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fill_random(float *M, int N) {
  for (int i = 0; i < N; i++)
    M[i] = (float)rand() / RAND_MAX;
}

// AttentionParams
void init_attention_params(AttentionParams *params, int d_model, int num_heads,
                           int random_init) {
  if (!params) {
    fprintf(stderr, "Error: failed to allocate AttentionParams\n");
    exit(1);
  }

  int d_k = d_model / num_heads;

  // Each head has its own W_qkv of shape (d_model x 3*d_k)
  // So total W_qkv = num_heads * d_model * 3*d_k
  size_t qkv_size = (size_t)d_model * 3 * d_k * num_heads;
  size_t wo_size = (size_t)d_model * d_model;

  params->W_qkv = malloc(sizeof(float) * qkv_size);
  params->W_o = malloc(sizeof(float) * wo_size);

  if (!params->W_qkv || !params->W_o) {
    fprintf(stderr, "Error: failed to allocate W_qkv or W_o\n");
    free_attention_params(params);
    exit(1);
  }

  // Initialize parameters
  for (size_t i = 0; i < qkv_size; i++) {
    params->W_qkv[i] =
        random_init ? ((float)rand() / RAND_MAX - 0.5f) * 0.1f : 0.0f;
  }

  for (size_t i = 0; i < wo_size; i++) {
    params->W_o[i] =
        random_init ? ((float)rand() / RAND_MAX - 0.5f) * 0.1f : 0.0f;
  }
}

void free_attention_params(AttentionParams *params) {
  if (!params)
    return;
  free(params->W_qkv);
  free(params->W_o);
  params->W_qkv = NULL;
  params->W_o = NULL;
}

// LayerNorm
void init_layernorm_params(LayerNormParams *params, int d_model) {
  if (!params) {
    fprintf(stderr, "Error: failed to allocate LayerNormParams");
    exit(1);
  }

  params->gamma = (float *)calloc(d_model, sizeof(float));
  params->beta = (float *)calloc(d_model, sizeof(float));

  if (!params->gamma || !params->beta) {
    fprintf(stderr, "Memory allocation failed for LayerNormParams.\n");
    free_layernorm_params(params);
    exit(1);
  }

  // Initialize gamma (scale) to 1.0, beta (bias) to 0.0 — standard defaults
  for (int i = 0; i < d_model; i++) {
    params->gamma[i] = 1.0f;
    params->beta[i] = 0.0f;
  }
}

void free_layernorm_params(LayerNormParams *params) {
  if (!params)
    return;
  free(params->gamma);
  free(params->beta);
  params->gamma = NULL;
  params->beta = NULL;
}

// FeedForward
void init_feedforward_params(FeedForwardParams *params, int d_model, int d_ff) {
  if (!params) {
    fprintf(stderr, "Error: NULL pointer passed in init_feedforward_params\n");
    exit(1);
  }

  params->W1 = (float *)malloc(d_model * d_ff * sizeof(float));
  params->B1 = (float *)calloc(d_ff, sizeof(float));
  params->W2 = (float *)malloc(d_ff * d_model * sizeof(float));
  params->B2 = (float *)calloc(d_model, sizeof(float));

  if (!params->W1 || !params->W2 || !params->B1 || !params->B2) {
    fprintf(stderr, "Memory allocation failed for FeedForwardParams");
    free_feedforward_params(params);
    exit(1);
  }

  float limit = sqrtf(6.0f / (d_model + d_ff));

  // Xavier G(lorot Initialization
  for (int i = 0; i < d_model * d_ff; i++)
    params->W1[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;

  for (int i = 0; i < d_ff * d_model; i++)
    params->W2[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
}

void free_feedforward_params(FeedForwardParams *params) {
  if (!params)
    return;
  free(params->W1);
  free(params->B1);
  free(params->W2);
  free(params->B2);
  params->W1 = params->W2 = params->B1 = params->B2 = NULL;
}

void init_encoder_params(EncoderLayerParams *params, int d_model, int d_ff,
                         int num_heads, int random_init) {
  if (!params) {
    fprintf(stderr, "Error: NULL pointer passed in init_feedforward_params\n");
    exit(1);
  }

  init_attention_params(&(params->attn_params), d_model, num_heads,
                        random_init);
  init_layernorm_params(&(params->ln1_params), d_model);
  init_feedforward_params(&(params->ffn_params), d_model, d_ff);
  init_layernorm_params(&(params->ln2_params), d_model);
}

void free_encoder_params(EncoderLayerParams *params) {
  if (!params)
    return;

  free_attention_params(&(params->attn_params));
  free_layernorm_params(&(params->ln1_params));
  free_feedforward_params(&(params->ffn_params));
  free_layernorm_params(&(params->ln2_params));
}

void init_decoder_layer_params(DecoderLayerParams *params, int d_model,
                               int d_ff, int num_heads, int random_init) {
  if (!params) {
    fprintf(stderr, "Error: NULL pointer passed in init_decoder_layer_params\n");
    exit(1);
  }

  init_attention_params(&(params->self_attn_params), d_model, num_heads,
                        random_init);
  init_layernorm_params(&(params->ln1_params), d_model);
  init_attention_params(&(params->cross_attn_params), d_model, num_heads,
                        random_init);
  init_layernorm_params(&(params->ln2_params), d_model);
  init_feedforward_params(&(params->ffn_params), d_model, d_ff);
  init_layernorm_params(&(params->ln3_params), d_model);
}

void free_decoder_layer_params(DecoderLayerParams *params) {
  if (!params)
    return;

  free_attention_params(&(params->self_attn_params));
  free_layernorm_params(&(params->ln1_params));
  free_attention_params(&(params->cross_attn_params));
  free_layernorm_params(&(params->ln2_params));
  free_feedforward_params(&(params->ffn_params));
  free_layernorm_params(&(params->ln3_params));
}

void init_transformer_params(TransformerParams *params,
                             TransformerConfig config) {
  params->config = config;

  // 1. Embeddings
  params->token_embedding =
      (float *)malloc(config.vocab_size * config.d_model * sizeof(float));
  params->pos_encoding =
      (float *)malloc(config.max_seq_len * config.d_model * sizeof(float));

  fill_random(params->token_embedding, config.vocab_size * config.d_model);
  fill_random(params->pos_encoding, config.max_seq_len * config.d_model);

  // 2. Encoder Layers
  params->encoder_layers = (EncoderLayerParams *)malloc(
      config.num_layers * sizeof(EncoderLayerParams));
  for (int i = 0; i < config.num_layers; i++) {
    init_encoder_params(&params->encoder_layers[i], config.d_model, config.d_ff,
                        config.num_heads, 1);
  }

  // 3. Decoder Layers
  params->decoder_layers = (DecoderLayerParams *)malloc(
      config.num_layers * sizeof(DecoderLayerParams));
  for (int i = 0; i < config.num_layers; i++) {
    init_decoder_layer_params(&params->decoder_layers[i], config.d_model,
                               config.d_ff, config.num_heads, 1);
  }

  // 4. Output Projection
  params->output_projection =
      (float *)malloc(config.d_model * config.vocab_size * sizeof(float));
  fill_random(params->output_projection, config.d_model * config.vocab_size);
}

void free_transformer_params(TransformerParams *params) {
  if (!params)
    return;

  free(params->token_embedding);
  free(params->pos_encoding);

  for (int i = 0; i < params->config.num_layers; i++) {
    free_encoder_params(&params->encoder_layers[i]);
    free_decoder_layer_params(&params->decoder_layers[i]);
  }
  free(params->encoder_layers);
  free(params->decoder_layers);
  free(params->output_projection);
}
