#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "decoder.h"
#include "encoder.h"

typedef struct {
  int num_layers;
  int d_model;
  int d_ff;
  int num_heads;
  int vocab_size;
  int max_seq_len;
} TransformerConfig;

typedef struct {
  TransformerConfig config;

  // 1. Embeddings
  float *token_embedding; // Shape: vocab_size x d_model
  float *pos_encoding;    // Shape: max_seq_len x d_model

  // 2. Encoder: Array of N layers
  EncoderLayerParams *encoder_layers;

  // 3. Decoder: Array of N layers
  DecoderLayerParams *decoder_layers;

  // 4. Final Output Projection
  float *output_projection; // Shape: d_model x vocab_size
} TransformerParams;

/**
 * @brief Forward pass for the complete Transformer.
 * @param src_input Tokens for the source sentence (Encoder input)
 * @param tgt_input Tokens for the target sentence (Decoder input)
 * @param out_logits Final probability distribution over vocab
 */
void compute_transformer(const int *src_tokens, const int *tgt_tokens,
                         const TransformerParams *params, float *out_logits,
                         int L_src, int L_tgt);

// Lifecycle functions
void init_transformer_params(TransformerParams *params,
                             TransformerConfig config);
void free_transformer_params(TransformerParams *params);

#endif
