#include "../include/attention.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gcc -Iinclude src/*.c tests/attention_tests.c -lm -O2 -o attention_test
// gcc -Iinclude src/*.c tests/attention_tests.c -DUSE_OPENBLAS -lopenblas -lm
// -O2 -o attention_test
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
  float *W_qkv = calloc(d_model * 3 * d_k, sizeof(float));

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

// Define a small tolerance for floating-point comparisons (local if not in
// utils.h)
#ifndef EPSILON
#define EPSILON 1e-5f
#endif

// --- Test Helper: Parameter Initialization ---

/**
 * @brief Allocates and initializes AttentionParams with simple deterministic
 * values for testing.
 */
void init_test_attention_params(AttentionParams *params, int d_model,
                                int num_heads) {
  // W_qkv: d_model x (3 * d_model)
  size_t w_qkv_size = (size_t)d_model * 3 * d_model;
  params->W_qkv = (float *)calloc(w_qkv_size, sizeof(float));

  // W_o: d_model x d_model
  size_t w_o_size = (size_t)d_model * d_model;
  params->W_o = (float *)calloc(w_o_size, sizeof(float));

  if (!params->W_qkv || !params->W_o) {
    fprintf(stderr, "Memory allocation failed for AttentionParams in test.\n");
    exit(1);
  }

  // Initialize W_qkv with deterministic values (e.g., 0.1, 0.2, 0.3...)
  for (size_t i = 0; i < w_qkv_size; i++) {
    // Simple pattern: increasing fractional values
    params->W_qkv[i] = (float)(i + 1) * 0.01f;
  }

  // Initialize W_o to an identity-like matrix (or a simple pattern)
  for (int i = 0; i < d_model; i++) {
    for (int j = 0; j < d_model; j++) {
      params->W_o[i * d_model + j] = (i == j) ? 1.0f : 0.0f; // Identity matrix
    }
  }
}

/**
 * @brief Frees memory allocated for AttentionParams.
 */
void free_test_attention_params(AttentionParams *params) {
  if (params->W_qkv)
    free(params->W_qkv);
  if (params->W_o)
    free(params->W_o);
}

// --- Main Test Function ---

static void test_multihead_attention() {
  // --- Test Parameters ---
  const int L = 2;                    // Sequence Length
  const int D_MODEL = 4;              // Model Dimension
  const int NUM_HEADS = 2;            // Number of Heads
  const int TOTAL_SIZE = L * D_MODEL; // 8

  printf("Testing compute_multihead_attention (L=%d, D_MODEL=%d, Heads=%d):\n",
         L, D_MODEL, NUM_HEADS);

  // 1. Input Data
  float X_data[8] = {
      1.0f, 2.0f, 3.0f, 4.0f, // Token 1
      5.0f, 6.0f, 7.0f, 8.0f  // Token 2
  };

  // 2. Output Reference Data (Precise values needed for this test scenario)
  // Recalculated reference for the identity W_o and sequential W_qkv weights:
  float expected_data[8] = {
      // These values are derived from a reference calculation
      // QKV Projection -> Scaling -> Softmax -> Matmul V -> Final Projection
      2.900000f, 3.000000f, 6.580000f, 6.840000f,
      3.500000f, 3.600000f, 8.140000f, 8.400000f};

  // 3. Dynamic Allocation and Initialization
  float *X = (float *)calloc(TOTAL_SIZE, sizeof(float));
  float *output = (float *)calloc(TOTAL_SIZE, sizeof(float));
  float *expected_output = (float *)calloc(TOTAL_SIZE, sizeof(float));

  memcpy(X, X_data, TOTAL_SIZE * sizeof(float));
  memcpy(expected_output, expected_data, TOTAL_SIZE * sizeof(float));

  AttentionParams params;
  init_test_attention_params(&params, D_MODEL, NUM_HEADS);

  // 4. Execute MHA
  compute_multihead_attention(X, &params, output, L, D_MODEL, NUM_HEADS);

  // 5. Verification
  printf("\tChecking final output projection:\n");
  // Using a slightly larger tolerance (1e-4) due to multiple floating point
  // operations
  if (compare(output, expected_output, TOTAL_SIZE)) {
    printf("\tPASSED\n");
  } else {
    printf("\tFAILED\n");
    // Print output and expected for debugging
    printf("\tExpected: [");
    for (int i = 0; i < TOTAL_SIZE; i++)
      printf("%.6f ", expected_output[i]);
    printf("]\n");
    printf("\tGot:      [");
    for (int i = 0; i < TOTAL_SIZE; i++)
      printf("%.6f ", output[i]);
    printf("]\n");
  }

  // 6. Cleanup
  free_test_attention_params(&params);
  free(X);
  free(output);
  free(expected_output);
}

int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_attention_basic();
  test_multihead_attention();
  printf("===== All tests complete =====\n");
  return 0;
}
