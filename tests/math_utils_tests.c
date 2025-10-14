#include "math_utils.h"

#include <cblas.h> // OpenBLAS header
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-3

void fill_random(float *M, int N) {
  for (int i = 0; i < N * N; i++)
    M[i] = (float)rand() / RAND_MAX;
}

int compare(float *C1, float *C2, int len) {
  for (int i = 0; i < len; i++) {
    if (fabsf(C1[i] - C2[i]) > EPSILON) {
      printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
      return 0;
    }
  }
  return 1;
}

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
  matmul_blocked(A, B, C_test, N);

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

int main(void) {
  // return 0 & 1 for those function
  matmul_blocked_test();

  transpose_matrix_test();

  return 0;
}
