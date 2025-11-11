#include "../include/init.h"

#include <stdio.h>

// gcc -Iinclude src/init.c src/utils.c -o init_tests tests/init_tests.c -lm -O2

static void print_mat(const char *name, const float *arr, int size,
                      int max_print) {
  printf("%s (size=%d): ", name, size);
  for (int i = 0; i < size && i < max_print; i++) {
    printf("%.4f ", arr[i]);
  }
  if (size > max_print)
    printf("...");
  printf("\n");
}

int main() {
  printf("===== Testing Parameter Initialization =====\n\n");

  // ---  AttentionParams Test ---
  printf("1 Testing AttentionParams initialization...\n");
  AttentionParams attn;
  init_attention_params(&attn, 4, 2,
                        1); // d_model=4, num_heads=2, random_init=1
  print_mat("W_qkv", attn.W_qkv, 4 * 3 * (4 / 2) * 2, 12);
  print_mat("W_o", attn.W_o, 4 * 4, 8);
  free_attention_params(&attn);
  printf("AttentionParams test complete.\n\n");

  // --- LayerNormParams Test ---
  printf("2 Testing LayerNormParams initialization...\n");
  LayerNormParams ln;
  init_layernorm_params(&ln, 4);
  print_mat("gamma", ln.gamma, 4, 4);
  print_mat("beta", ln.beta, 4, 4);
  free_layernorm_params(&ln);
  printf("LayerNormParams test complete.\n\n");

  // --- FeedForwardParams Test ---
  printf("3 Testing FeedForwardParams initialization...\n");
  FeedForwardParams ff;
  init_feedforward_params(&ff, 4, 8); // d_model=4, d_ff=8
  print_mat("W1", ff.W1, 4 * 8, 8);
  print_mat("B1", ff.B1, 8, 8);
  print_mat("W2", ff.W2, 8 * 4, 8);
  print_mat("B2", ff.B2, 4, 4);
  free_feedforward_params(&ff);
  printf("FeedForwardParams test complete.\n\n");

  printf("===== All initialization tests passed. =====\n");
  return 0;
}
