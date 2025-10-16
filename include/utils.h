// linear algebra helpers

void matmul_blocked(const float *A, const float *B, float *C, int M, int N,
                    int K);
void transpose_matrix(const float *src, float *dst, int rows, int cols);

// misc utilities
#ifndef SOFTMAX_H
#define SOFTMAX_H

void scale_scores(float *scores, int L, int d_k);
void apply_mask(float *scores, const float *mask, int L);
void softmax_rows(const float *scores, float *weights, int L);

#endif

int compare(float *C1, float *C2, int len);

void fill_random(float *M, int N);
void mattri_low(float *A, int M);
