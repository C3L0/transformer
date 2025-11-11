#include "../include/layernorm.h"
#include "../include/utils.h"

#include <stdio.h>

// gcc -Iinclude src/*.c tests/layernorm_tests.c -o layernorm_tests -lm

static void test_full_layernorm() {
  int L = 1;
  int d_model = 4;

  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float output[4];

  LayerNormParams params;
  float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  params.gamma = gamma;
  params.beta = beta;

  // Reference Calculation:
  // Mean = 2.5
  // Variance = 1.25
  // Inv Std Dev = 1 / sqrt(1.25) ~ 0.894423
  // x_hat: (-1.5 * 0.894423), (-0.5 * 0.894423), (0.5 * 0.894423), (1.5 *
  // 0.894423)
  float ref[4] = {-1.341639f, -0.447213f, 0.447213f, 1.341639f};

  // Compute
  compute_layernorm(input, &params, output, L, d_model);

  // Compare
  printf("Testing full compute_layernorm (d_model=%d):\n", d_model);
  if (compare(output, ref, d_model)) {
    printf("\tPASSED\n");
  } else {
    printf("\tFAILED\n");
  }
}

int main() {
  test_full_layernorm();
  return 0;
}
