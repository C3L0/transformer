#include "math_utils.h"

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 64 // depends on your CPU cache size

// matrix product not optimize
void matmul_blocked(float *A, float *B, float *C, int N) {
  for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < N; kk += BLOCK_SIZE) {

        int i_max = (ii + BLOCK_SIZE > N) ? N : ii + BLOCK_SIZE;
        int j_max = (jj + BLOCK_SIZE > N) ? N : jj + BLOCK_SIZE;
        int k_max = (kk + BLOCK_SIZE > N) ? N : kk + BLOCK_SIZE;

        // Multiply the blocks
        for (int i = ii; i < i_max; i++) {
          for (int k = kk; k < k_max; k++) {
            float a_ik = A[i * N + k];
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
