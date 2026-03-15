#include <stdio.h>
#include <stdlib.h>
#include "include/transformer.h"

int main(void) {
  printf("Transformer Model Demo\n");

  // 1. Model Configuration
  TransformerConfig config = {
      .num_layers = 2,
      .d_model = 64,
      .d_ff = 256,
      .num_heads = 4,
      .vocab_size = 1000,
      .max_seq_len = 128
  };

  // 2. Initialize Parameters
  TransformerParams params;
  init_transformer_params(&params, config);
  printf("Model initialized with %d layers and d_model=%d\n", config.num_layers,
         config.d_model);

  // 3. Prepare Dummy Input
  int L_src = 10;
  int L_tgt = 8;
  int *src_tokens = malloc(L_src * sizeof(int));
  int *tgt_tokens = malloc(L_tgt * sizeof(int));

  for (int i = 0; i < L_src; i++) src_tokens[i] = i % config.vocab_size;
  for (int i = 0; i < L_tgt; i++) tgt_tokens[i] = (i + 5) % config.vocab_size;

  // 4. Output Buffer
  float *out_logits = (float *)malloc(L_tgt * config.vocab_size * sizeof(float));

  // 5. Forward Pass
  printf("Running forward pass for %d source and %d target tokens...\n", L_src, L_tgt);
  compute_transformer(src_tokens, tgt_tokens, &params, out_logits, L_src, L_tgt);
  printf("Forward pass completed successfully.\n");

  // 6. Cleanup
  free(src_tokens);
  free(tgt_tokens);
  free(out_logits);
  free_transformer_params(&params);

  return 0;
}
