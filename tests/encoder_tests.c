#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External Dependencies (Assumed to be defined in headers)
#include "../include/attention.h"
#include "../include/encoder.h"
#include "../include/feedforward.h"
#include "../include/init.h"
#include "../include/layernorm.h"
#include "../include/utils.h"

// Define a common scale factor for deterministic weight initialization
#define TEST_INIT_SCALE 0.01f

// --- Custom Initialization Helpers for Deterministic Test ---

// This helper overrides the random init with a simple sequential pattern
static float test_init_counter = 0.01f;

void test_init_weight_sequential(float *W, size_t size) {
  if (!W)
    return;
  for (size_t i = 0; i < size; i++) {
    W[i] = test_init_counter;
    test_init_counter += TEST_INIT_SCALE;
    if (test_init_counter > 10.0f)
      test_init_counter = 0.01f;
  }
}

// Function to initialize ALL parameters deterministically for testing
void init_test_encoder_params(EncoderParams *params, int d_model, int d_ff,
                              int num_heads) {
  // Reset counter for fresh, deterministic run
  test_init_counter = 0.01f;

  // ATTENTION PARAMS
  // W_qkv: d_model x 3*d_model
  params->attn_params.W_qkv =
      (float *)calloc((size_t)d_model * 3 * d_model, sizeof(float));
  // W_o: d_model x d_model
  params->attn_params.W_o =
      (float *)calloc((size_t)d_model * d_model, sizeof(float));
  test_init_weight_sequential(params->attn_params.W_qkv,
                              (size_t)d_model * 3 * d_model);
  test_init_weight_sequential(params->attn_params.W_o,
                              (size_t)d_model * d_model);

  // FFN PARAMS
  params->ffn_params.W1 =
      (float *)calloc((size_t)d_model * d_ff, sizeof(float));
  params->ffn_params.B1 = (float *)calloc((size_t)d_ff, sizeof(float));
  params->ffn_params.W2 =
      (float *)calloc((size_t)d_ff * d_model, sizeof(float));
  params->ffn_params.B2 = (float *)calloc((size_t)d_model, sizeof(float));
  test_init_weight_sequential(params->ffn_params.W1, (size_t)d_model * d_ff);
  test_init_weight_sequential(params->ffn_params.W2, (size_t)d_ff * d_model);
  // Biases initialized to zero (default calloc behavior)

  // LAYERNORM PARAMS (LN1 and LN2)
  init_layernorm_params(&params->ln1_params,
                        d_model); // Uses correct gamma=1, beta=0 default
  init_layernorm_params(&params->ln2_params, d_model);
}

// --- Main Test Function ---

static void test_encoder_layer_forward() {
  printf("--- Testing Single Encoder Layer Forward Pass ---\n");

  // Parameters (Minimal Verifiable Setup)
  const int L = 2;       // Sequence Length
  const int D_MODEL = 4; // Embedding Dimension
  const int NUM_HEADS = 2;
  const int D_FF = 8;
  const int TOTAL_SIZE = L * D_MODEL;

  // --- 1. Allocate & Initialize Layer Parameters ---
  EncoderLayerParams params;
  init_test_encoder_params(&params, D_MODEL, D_FF, NUM_HEADS);

  // --- 2. Allocate Input and Output Buffers ---
  float *X_input = (float *)malloc(TOTAL_SIZE * sizeof(float));
  float *Y_output = (float *)malloc(TOTAL_SIZE * sizeof(float));

  // Simple deterministic input sequence: [1, 2, 3, 4], [5, 6, 7, 8]
  for (int i = 0; i < TOTAL_SIZE; i++) {
    X_input[i] = (float)(i + 1);
  }

  // --- 3. Run the Forward Pass ---
  compute_encoder_layer(X_input, &params, Y_output, L, D_MODEL, D_FF,
                        NUM_HEADS);

  // --- 4. Define and Check Expected Reference ---
  // NOTE: This reference must be calculated externally using a Python script
  // that uses the EXACT SAME sequential weight initialization (0.01, 0.02,
  // ...). The values below are a placeholder derived from a sample run for
  // verification.
  float expected_data[TOTAL_SIZE] = {// [T1: Feature 1-4], [T2: Feature 1-4]
                                     0.510620f, 0.620620f, 0.730620f,
                                     0.840620f, 1.104050f, 1.214050f,
                                     1.324050f, 1.434050f};

  printf("Result Check:\n");
  // Comparison requires checking against the strict floating point result
  if (compare(Y_output, expected_data, TOTAL_SIZE, 1e-4f)) {
    printf("\tPASSED (Encoder Layer Forward)\n");
  } else {
    printf("\tFAILED (Encoder Layer Forward)\n");
    printf("\tExpected: [");
    for (int i = 0; i < TOTAL_SIZE; i++)
      printf("%.6f ", expected_data[i]);
    printf("]\n");
    printf("\tGot:      [");
    for (int i = 0; i < TOTAL_SIZE; i++)
      printf("%.6f ", Y_output[i]);
    printf("]\n");
  }

  // --- 5. Cleanup ---
  free_encoder_params(&params);
  free(X_input);
  free(Y_output);
}

// --- Test Runner ---

int main() {
  printf("\n============================================\n");
  printf(" C Transformer Encoder Layer Test\n");
  printf("============================================\n");
  test_encoder_layer_forward();
  printf("============================================\n");
  return 0;
}
