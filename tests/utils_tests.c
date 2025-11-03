#include "../include/utils.h"

#include <cblas.h> // OpenBLAS header
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gcc -Iinclude src/utils.c tests/utils_tests.c -o utils_tests -lopenblas -lm
// -O2

void test_matmul_blocked(void) {
  int N = 256; // test matrix size

  float *A = calloc(N * N, sizeof(float));
  float *B = calloc(N * N, sizeof(float));
  float *C_ref = calloc(N * N, sizeof(float));
  float *C_test = calloc(N * N, sizeof(float));

  fill_random(A, N);
  fill_random(B, N);

  // Reference result (OpenBLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B,
              N, 0.0f, C_ref, N);

  // My implementation
  matmul_blocked(A, B, C_test, N, N, N); // need to be checked

  printf("Testing matmul_blocked vs OpenBLAS SGEMM (N=%d):\n\t", N);
  if (compare(C_ref, C_test, N * N))
    printf("PASSED\n");
  else
    printf("FAILED\n");

  free(A);
  free(B);
  free(C_ref);
  free(C_test);
}

void test_transpose_matrix(void) {
  int cols = 3, rows = 2;

  float A[6] = {1, 2, 3, 4, 5, 6};
  float A_T[6] = {0};
  float A_T_ref[6] = {1, 4, 2, 5, 3, 6};

  transpose_matrix(A, A_T, rows, cols);

  printf("Testing transpose_matrix_test:\n\t");
  if (compare(A_T_ref, A_T, cols * rows))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

#define EPSILON 1e-5

static int float_equal(float a, float b) { return fabsf(a - b) < EPSILON; }

static void test_scale_scores() {
  float scores[4] = {1.0, 2.0, 3.0, 4.0};
  int L = 2, d_k = 4;

  float ref[4] = {0.5, 1.0, 1.5, 2.0};

  scale_scores(scores, L, d_k);

  printf("Testing scale_scores:\n\t");
  if (compare(scores, ref, 4))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

static void test_apply_mask() {
  float scores[4] = {1.0, 2.0, 3.0, 4.0};
  float mask[4] = {1.0, 0.0, 1.0, 0.0};
  int L = 2;

  float ref[4] = {1.0, -INFINITY, 3.0, -INFINITY};

  apply_mask(scores, mask, L);

  printf("Testing apply_mask:\n\t");
  if (compare(scores, ref, 4))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

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

static void test_mattri_low() {

  float *M = malloc(9 * sizeof(float));
  float M_ref[9] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0};

  mattri_low(M, 3);

  printf("Testing mattri_low:\n\t");
  if (compare(M, M_ref, 9))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

static void test_masking() {
  int L = 3;
  float scores[9];
  float mask[9];

  for (int i = 0; i < 9; i++)
    scores[i] = i;

  float ref[9] = {0.0,       -INFINITY, -INFINITY, 3.0, 4.0,
                  -INFINITY, 6.0,       7.0,       8.0};

  mattri_low(mask, L);
  apply_mask(scores, mask, L);

  printf("Masked scores:\n\t");
  if (compare(scores, ref, 9))
    printf("PASSED\n");
  else
    printf("FAILED\n");
}

static void test_matrix_add_vector_bias() {

  float input_matrix[12] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  float *M_test = malloc(12 * sizeof(float));
  memcpy(M_test, input_matrix, 12 * sizeof(float));

  float bias_vector[4] = {10.0f, 20.0f, 30.0f, 40.0f};

  float M_ref[12] = {11.0f, 22.0f, 33.0f, 44.0f, 15.0f, 26.0f,
                     37.0f, 48.0f, 19.0f, 30.0f, 41.0f, 52.0f};

  //  matrix_add_vector_bias(M_test, bias_vector, 3, 4);
  matrix_add_vector_bias(M_test, bias_vector, 3, 4);

  printf("Testing matrix_add_vector_bias:\n\t");
  if (compare(M_test, M_ref, 12)) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  free(M_test);
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

int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_matmul_blocked();
  test_transpose_matrix();
  test_scale_scores();
  test_apply_mask();
  test_softmax_rows();
  test_mattri_low();
  test_masking();
  test_matrix_add_vector_bias();
  test_apply_gelu();
  test_mean_variance();
  printf("===== All tests complete =====\n");
  return 0;
}
