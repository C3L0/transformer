#include "../include/transformer.h"
#include "../include/tensor.h"
#include "../include/init.h"
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include <stdlib.h>
#include <string.h>

/**
 * @brief Simple embedding lookup helper.
 */
static void apply_embedding(const int *tokens, int L, int d_model,
                            const float *emb_table, const float *pos_table,
                            float *out) {
  for (int i = 0; i < L; i++) {
    int token_id = tokens[i];
    // Copy token embedding
    memcpy(out + (i * d_model), emb_table + (token_id * d_model),
           d_model * sizeof(float));
    // Add positional encoding (Residual style)
    for (int d = 0; d < d_model; d++) {
      out[i * d_model + d] += pos_table[i * d_model + d];
    }
  }
}

void compute_transformer(const int *src_tokens, const int *tgt_tokens,
                         const TransformerParams *params, float *out_logits,
                         int L_src, int L_tgt) {

  int d_model = params->config.d_model;
  int d_ff = params->config.d_ff;
  int num_heads = params->config.num_heads;
  int num_layers = params->config.num_layers;

  // --- 1. Encoder Path ---
  float *enc_buf = (float *)calloc(L_src * d_model, sizeof(float));
  float *enc_input = (float *)calloc(L_src * d_model, sizeof(float));

  // Embedding + Positional Encoding
  apply_embedding(src_tokens, L_src, d_model, params->token_embedding,
                  params->pos_encoding, enc_input);

  // Iterative Encoder Layers
  float *current_src = enc_input;
  float *next_src = enc_buf;
  for (int i = 0; i < num_layers; i++) {
    compute_encoder_layer(current_src, &params->encoder_layers[i], next_src,
                          L_src, d_model, d_ff, num_heads);
    // Swap
    float *tmp = current_src;
    current_src = next_src;
    next_src = tmp;
  }
  // Result: current_src now contains the final Encoder context

  // --- 2. Decoder Path ---
  float *dec_buf = (float *)calloc(L_tgt * d_model, sizeof(float));
  float *dec_input = (float *)calloc(L_tgt * d_model, sizeof(float));

  // Embedding + Positional Encoding
  apply_embedding(tgt_tokens, L_tgt, d_model, params->token_embedding,
                  params->pos_encoding, dec_input);

  float *current_tgt = dec_input;
  float *next_tgt = dec_buf;
  for (int i = 0; i < num_layers; i++) {
    // Note: Cross-attention always uses the final current_src from the Encoder
    compute_decoder_layer(current_tgt, current_src, &params->decoder_layers[i],
                          next_tgt, L_tgt, L_src, d_model, d_ff, num_heads);
    // Swap
    float *tmp = current_tgt;
    current_tgt = next_tgt;
    next_tgt = tmp;
  }

  // --- 3. Final Output Projection (Logits) ---
  // (L_tgt x d_model) * (d_model x vocab_size) = (L_tgt x vocab_size)
  // We use the last current_tgt (decoder output)
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L_tgt,
              params->config.vocab_size, d_model, 1.0f, current_tgt, d_model,
              params->output_projection, params->config.vocab_size, 0.0f,
              out_logits, params->config.vocab_size);
#else
  matmul_blocked(current_tgt, params->output_projection, out_logits, L_tgt,
                 params->config.vocab_size, d_model);
#endif

  // Cleanup
  free(enc_buf);
  free(enc_input);
  free(dec_buf);
  free(dec_input);
}
