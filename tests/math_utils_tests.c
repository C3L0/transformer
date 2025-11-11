#include "../include/math_utils.h"
#include "../include/utils.h"

#include <math.h>
#include <stdio.h>

// gcc -Iinclude src/math_utils.c src/utils.c tests/math_utils_tests.c -o
// math_utils_tests -lm -O2

static void test_softmax_rows() {

  float scores[4] = {0.0, 1.0, 2.0, 3.0}; // 2x2
  float ref[4] = {0.26894143, 0.7310586, 0.26894143, 0.7310586};
  float weights[4];
  int L = 2;

  softmax_rows(scores, weights, L);

  printf("Testing softmax_rows:\n\t");
  if (compare(weights, ref, 4))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

static void test_mean_variance() {

  float mat[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
  float mean = 0.0;
  float var = 0.0;

  float mean_ref = 2.0;
  float var_ref = 2.0;

  compute_mean_variance(mat, 5, &mean, &var);

  printf("Testing mean and variance calculation:\n\t");
  if (fabsf(mean - mean_ref) < EPSILON && fabsf(var - var_ref) < EPSILON) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

static void test_apply_gelu() { // need to be check a better way but with python
                                // the result is good
  float mat[3] = {-1.0f, 0.0f, 1.0f};
  apply_gelu(mat, 1, 3);
  // result from a python script
  float mat_ref[3] = {-0.1588f, 0.0f, 0.8412};

  printf("Testing matrix_add_vector_bias:\n\t");
  if (compare(mat, mat_ref, 3)) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
}

int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_softmax_rows();
  test_apply_gelu();
  test_mean_variance();
  printf("===== All tests complete =====\n");
  return 0;
}
