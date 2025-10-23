#include "../include/attention.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gcc -Iinclude src/*.c tests/attention_tests.c -lm -O2 -o attention_test
// ./attention_test

// small helper to print
static void print_mat(const char *name, const float *A, int R, int C) {
  printf("%s (%dx%d):\n", name, R, C);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      printf("%8.4f ", A[i * C + j]);
    }
    printf("\n");
  }
}

void test_attention_basic() {

  int L = 2;
  int d_model = 2;
  int d_k = 2;

  // Input X (L x d_model)
  float X[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  // Concatenated weights W_qkv = [W_Q | W_K | W_V]
  // Each W_* is (d_model x d_k)
  float W_Q[4] = {0.1f, 0.2f, 0.3f, 0.4f};

  float W_K[4] = {0.5f, 0.6f, 0.7f, 0.8f};

  float W_V[4] = {0.9f, 1.0f, 1.1f, 1.2f};

  // Concatenate into one big W_qkv (d_model x 3*d_k)
  float W_qkv[12];
  for (int i = 0; i < d_model; i++) {
    for (int j = 0; j < d_k; j++) {
      W_qkv[i * (3 * d_k) + j + 0 * d_k] = W_Q[i * d_k + j];
      W_qkv[i * (3 * d_k) + j + 1 * d_k] = W_K[i * d_k + j];
      W_qkv[i * (3 * d_k) + j + 2 * d_k] = W_V[i * d_k + j];
    }
  }

  float Q[4], K[4], V[4];
  float scores[4];
  float weights[4];
  float out[4];

  compute_attention_gemm(X, W_qkv, Q, K, V, scores, weights, out, L, d_model,
                         d_k);

  // print results
  print_mat("out", out, L, d_k);
  return;
}
int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_attention_basic();
  printf("===== All tests complete =====\n");
  return 0;
}
