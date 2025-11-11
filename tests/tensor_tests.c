#include "../include/init.h"
#include "../include/tensor.h"
#include "../include/utils.h"

#include <cblas.h>
#include <stdlib.h>
#include <string.h>

// gcc -Iinclude src/tensor.c src/utils.c src/init.c tests/tensor_tests.c -o
// tensor_tests -lopenblas -lm -O2

static void test_matmul_blocked(void) {
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

static void test_transpose_matrix(void) {
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

int main() {
  // return 0 & 1 for the tests
  printf("===== Running utils unit tests =====\n");
  test_matmul_blocked();
  test_transpose_matrix();
  test_mattri_low();
  test_matrix_add_vector_bias();
  printf("===== All tests complete =====\n");
  return 0;
}
