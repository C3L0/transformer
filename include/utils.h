// linear algebra helpers

void matmul_blocked(const float *A, const float *B, float *C, int M, int N,
                    int K);
void transpose_matrix(const float *src, float *dst, int rows, int cols);
void matrix_add_vector_bias(float *matrix, const float *bias, int M, int N);

// misc utilities
#ifndef SOFTMAX_H
#define SOFTMAX_H

void scale_scores(float *scores, int L, int d_k);
void apply_mask(float *scores, const float *mask, int L);
void softmax_rows(const float *scores, float *weights, int L);
void compute_mean_variance(const float *M, int size, float *mean_out,
                           float *var_out);

#define GELU_A 0.044715f
#define SQRT_2_OVER_PI 0.7978845608f // sqrt(2/pi)
void apply_gelu(float *M, int rows, int cols);

#endif

int compare(float *C1, float *C2, int len);

void fill_random(float *M, int N);
void mattri_low(float *A, int M);
