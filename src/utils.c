#include "../include/utils.h"

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

#define BLOCK_SIZE 64 // depends on your CPU cache size

void matmul_blocked(const float *A, const float *B, float *C, int M, int N,
                    int K) {
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

void scale_scores(float *scores, int L, int d_k) {
  float scale = 1.0f / sqrtf((float)d_k);
  for (int i = 0; i < L * L; i++)
    scores[i] *= scale;
}

void apply_mask(float *scores, const int *mask, int L) {
  if (!mask)
    return;
  for (int i = 0; i < L * L; i++)
    if (mask[i] == 0)
      scores[i] = -INFINITY;
}

void softmax_rows(const float *scores, float *weights, int L) {
  for (int i = 0; i < L; i++) {
    float max_val = -INFINITY, sum = 0.0f;
    for (int j = 0; j < L; j++) {
      float val = scores[i * L + j];
      if (val > max_val)
        max_val = val;
    }
    for (int j = 0; j < L; j++) {
      float e = expf(scores[i * L + j] - max_val);
      weights[i * L + j] = e;
      sum += e;
    }
    for (int j = 0; j < L; j++)
      weights[i * L + j] /= sum;
  }
}
