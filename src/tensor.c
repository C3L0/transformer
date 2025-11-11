#include "../include/tensor.h"

#include <string.h>

void mattri_low(float *A, int M) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      A[i * M + j] = (j <= i) ? 1.0 : 0.0;
    }
  }
}

#define BLOCK_SIZE 64 // depends on your CPU cache size

void matmul_blocked(const float *A, const float *B, float *C, int M, int N,
                    int K) {

  memset(C, 0, M * N * sizeof(float));
  // A: M×K,  B: K×N,  C: M×N
  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < K; kk += BLOCK_SIZE) {

        int i_max = (ii + BLOCK_SIZE > M) ? M : ii + BLOCK_SIZE;
        int j_max = (jj + BLOCK_SIZE > N) ? N : jj + BLOCK_SIZE;
        int k_max = (kk + BLOCK_SIZE > K) ? K : kk + BLOCK_SIZE;

        for (int i = ii; i < i_max; i++) {
          for (int k = kk; k < k_max; k++) {
            float a_ik = A[i * K + k];
            for (int j = jj; j < j_max; j++) {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

void transpose_matrix(const float *src, float *dst, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}

void matrix_add_vector_bias(float *matrix, const float *bias, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i * N + j] += bias[j];
    }
  }
}
