#include "../include/utils.h"

#include <cblas.h> // OpenBLAS header
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// gcc -Iinclude src/utils.c tests/utils_tests.c -o utils_tests -lopenblas -lm
// -O2

void matmul_blocked_test(void) {
  int N = 256; // test matrix size
  printf("Testing matmul_blocked vs OpenBLAS SGEMM (N=%d): ", N);

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

  if (compare(C_ref, C_test, N * N))
    printf("PASSED — Results match!\n");
  else
    printf("FAILED — Results differ!\n");

  free(A);
  free(B);
  free(C_ref);
  free(C_test);
}

void transpose_matrix_test(void) {
  int cols = 3, rows = 2;

  float A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
  float A_T[6] = {0};
  float A_T_ref[6] = {1, 4, 2, 5, 3, 6};

  // My implementation
  transpose_matrix(A, A_T, rows, cols);

  printf("Testing transpose_matrix_test: ");
  if (compare(A_T_ref, A_T, cols * rows))
    printf("PASSED — Results match!\n");
  else
    printf("FAILED — Results differ!\n");
}

#define EPSILON 1e-5

static int float_equal(float a, float b) { return fabsf(a - b) < EPSILON; }

static void test_scale_scores() {
  printf("Testing test_scale_scores: ");
  float scores[4] = {1.0, 2.0, 3.0, 4.0};
  int L = 2, d_k = 4;

  float ref[4] = {0.5, 1.0, 1.5, 2.0};

  scale_scores(scores, L, d_k);

  float expected_scale = 1.0f / sqrtf(4.0f);
  int pass = 1;
  for (int i = 0; i < 4; i++) {
    if (!float_equal(scores[i], (i + 1) * expected_scale)) {
      printf("FAILED — Results differ!\n");
      printf("scale_scores failed at index %d: got %f expected %f\n", i,
             scores[i], (i + 1) * expected_scale);
      pass = 0;
    }
  }
  if (pass)
    printf("PASSED — Results match!\n");
}

static void test_apply_mask() {
  printf("Testing test_apply_mask: ");
  float scores[4] = {1.0, 2.0, 3.0, 4.0};
  int mask[4] = {1, 0, 1, 0};
  int L = 2;

  apply_mask(scores, mask, L);

  int pass = 1;
  for (int i = 0; i < 4; i++) {
    if (mask[i] == 0) {
      if (!isinf(scores[i])) {
        printf("FAILED — Results differ!\n");
        printf("apply_mask failed at index %d: expected -inf got %f\n", i,
               scores[i]);
        pass = 0;
      }
    }
  }
  if (pass)
    printf("PASSED — Results match!\n");
}

static void test_softmax_rows() {
  printf("Testing test_softmax_rows: ");

  float scores[4] = {0.0, 1.0, 2.0, 3.0}; // 2x2
  float ref[4] = {0.26894143, 0.7310586, 0.26894143, 0.7310586};
  float weights[4];
  int L = 2;

  softmax_rows(scores, weights, L);

  int pass = 1;
  for (int i = 0; i < 4; i++) {
    if (!float_equal(weights[i], ref[i])) {
      printf("FAILED — Results differ!\n");
      printf("softmax_rows[%d]: got %f, expected %f\n", i, weights[i], ref[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("PASSED — Results match!\n");
}

int main() {

  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  matmul_blocked_test();
  transpose_matrix_test();
  test_scale_scores();
  test_apply_mask();
  test_softmax_rows();
  printf("===== All tests complete =====\n");
  return 0;
}
