#include "../include/feedforward.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>

/// NOTE improve the precision for the entire code maybe work on the type
///  int->size_t or float->double
static void test_compute_feedforward_network() {
  const int L = 1;
  const int D_MODEL = 2;
  const int D_FF = 4;

  // Input vector (1 x 2)
  float X[2] = {1.0f, 2.0f};

  // Initialize simple, deterministic weights and biases
  FeedForwardParams params;

  // W1: (2 x 4)
  params.W1 = (float *)malloc(D_MODEL * D_FF * sizeof(float));
  for (int i = 0; i < D_MODEL * D_FF; i++)
    params.W1[i] = 0.1f;
  params.W1[0] = 1.0f;
  params.W1[1] = 0.0f;
  params.W1[2] = 0.5f;
  params.W1[3] = 0.5f;
  params.W1[4] = 0.0f;
  params.W1[5] = 1.0f;
  params.W1[6] = 0.5f;
  params.W1[7] = 0.5f;

  // B1: (1 x 4)
  params.B1 = (float *)malloc(D_FF * sizeof(float));
  for (int i = 0; i < D_FF; i++)
    params.B1[i] = 1.0f;

  // W2: (4 x 2)
  params.W2 = (float *)malloc(D_FF * D_MODEL * sizeof(float));
  for (int i = 0; i < D_FF * D_MODEL; i++)
    params.W2[i] = 0.5f;

  // B2: (1 x 2)
  params.B2 = (float *)malloc(D_MODEL * sizeof(float));
  for (int i = 0; i < D_MODEL; i++)
    params.B2[i] = 2.0f;

  // Output buffer (1 x 2)
  float out[2];

  float out_ref[2] = {6.97695f, 6.97695f};

  // Execute the FFNN forward pass
  compute_feedforward_network(X, &params, out, L, D_MODEL, D_FF);

  printf("Testing full compute_feedforward_network:\n\t");
  if (compare(out, out_ref, D_MODEL))
    printf("PASSED\n");
  else
    printf("FAILED (Expected: %.5f, %.5f | Got: %.5f, %.5f)\n", out_ref[0],
           out_ref[1], out[0], out[1]);

  free(params.W1);
  free(params.B1);
  free(params.W2);
  free(params.B2);
}

// --- EXPECTED CALCULATION ---
// H1_pre_act = (X @ W1) + B1
// X @ W1 = [1, 2] @ [[1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5]] = [1, 2, 1.5, 1.5]
// H1_pre_act = [1, 2, 1.5, 1.5] + [1, 1, 1, 1] = [2, 3, 2.5, 2.5]

// H1 = GELU(H1_pre_act)
// GELU(2.0) ~= 1.9546
// GELU(3.0) ~= 2.9997
// GELU(2.5) ~= 2.4998
// H1 ~= [1.9546, 2.9997, 2.4998, 2.4998]

// Out_pre_bias = H1 @ W2 (W2 is all 0.5)
// Out_pre_bias[0] = 0.5 * (1.9546 + 2.9997 + 2.4998 + 2.4998) ~= 0.5 * 9.9539
// = 4.97695 Out_pre_bias[1] = Out_pre_bias[0] ~= 4.97695 (due to symmetric
// W2)

// Out = Out_pre_bias + B2 ([2, 2])
// Out ~= [4.97695 + 2, 4.97695 + 2] = [6.97695, 6.97695]

int main() {
  printf("===== FFNN Component Tests =====\n");
  test_compute_feedforward_network();
  printf("===== test completed =====\n");
  return 0;
}
