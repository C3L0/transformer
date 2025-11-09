#include "../include/init.h"

#include "../include/attention.h"
#include "../include/feedforward.h"
#include "../include/layernorm.h"

#include "../include/utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    free(params->W_qkv);
    free(params->W_o);
    free(params);
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
  if (params) {
    free(params->W_qkv);
    free(params->W_o);
    free(params);
  }
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
    free(params->gamma);
    free(params->beta);
    exit(1);
  }

  // Initialize gamma (scale) to 1.0, beta (bias) to 0.0 — standard defaults
  for (int i = 0; i < d_model; i++) {
    params->gamma[i] = 1.0f;
    params->beta[i] = 0.0f;
  }
}

// Free LayerNorm parameters
void free_layernorm_params(LayerNormParams *params) {
  if (!params)
    return;
  free(params->gamma);
  free(params->beta);
  params->gamma = NULL;
  params->beta = NULL;
}

void init_feedforward_params() {}

void free_feedforward_params(FeedForwardParams *params) {}
