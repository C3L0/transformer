#include "../include/attention.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>

// gcc -Iinclude src/*.c tests/attention_tests.c -lm -O2 -o attention_test
// ./attention_test

void test_attention_basic() {
  int L = 2, d_model = 4, d_k = 2;

  float X[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  float W_qkv[24] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3,
                     0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.6,
                     0.7, 0.8, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  float Q[4], K[4], V[4], scores[4], weights[4], out[4];

  compute_attention_gemm(X, W_qkv, Q, K, V, scores, weights, out, L, d_model,
                         d_k);

  printf("Output:\n");
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < d_k; j++) {
      printf("%6.3f ", out[i * d_k + j]);
    }
    printf("\n");
  }
}
int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_attention_basic();
  printf("===== All tests complete =====\n");
  return 0;
}
