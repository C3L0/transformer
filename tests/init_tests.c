#include "../include/init.h"
#include "../include/utils.h"

#include <stdio.h>

// gcc -Iinclude src/init.c src/utils.c -o init_tests tests/init_tests.c -lm -O2

int main() {
  printf("===== Testing Parameter Initialization =====\n\n");

  // ---  AttentionParams Test ---
  printf("Testing AttentionParams initialization\n");
  AttentionParams attn;
  init_attention_params(&attn, 4, 2,
                        1); // d_model=4, num_heads=2, random_init=1
  print_mat("W_qkv", attn.W_qkv, 4 * 3 * (4 / 2) * 2, 12);
  print_mat("W_o", attn.W_o, 4 * 4, 8);
  free_attention_params(&attn);
  printf("AttentionParams test complete\n\n");

  // --- LayerNormParams Test ---
  printf("Testing LayerNormParams initialization\n");
  LayerNormParams ln;
  init_layernorm_params(&ln, 4);
  print_mat("gamma", ln.gamma, 4, 4);
  print_mat("beta", ln.beta, 4, 4);
  free_layernorm_params(&ln);
  printf("LayerNormParams test complete\n\n");

  // --- FeedForwardParams Test ---
  printf("Testing FeedForwardParams initialization\n");
  FeedForwardParams ff;
  init_feedforward_params(&ff, 4, 8); // d_model=4, d_ff=8
  print_mat("W1", ff.W1, 4 * 8, 8);
  print_mat("B1", ff.B1, 8, 8);
  print_mat("W2", ff.W2, 8 * 4, 8);
  print_mat("B2", ff.B2, 4, 4);
  free_feedforward_params(&ff);
  printf("FeedForwardParams test complete\n\n");

  // --- EncoderParams Test ---
  printf("Testing EncoderParams initialization\n");
  EncoderParams encode;
  init_encoder_params(&encode, 4, 8, 2, 1);
  // print_mat("AttentionParams", encode. , , );
  // print_mat("LayerNormParams", encode. , , );
  // print_mat("", encode. , , );
  // print_mat("", encode. , , );
  free_encoder_params(&encode);
  printf("===== All initialization tests passed. =====\n");
  return 0;
}
