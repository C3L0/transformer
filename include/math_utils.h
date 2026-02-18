// math functions

#ifndef SOFTMAX_H
#define SOFTMAX_H

#define EPSILON 1e-5

#define GELU_A 0.044715f
#define SQRT_2_OVER_PI 0.7978845608f

// void softmax_rows(const float *scores, float *weights, int L);
void softmax_rows(const float *scores, float *weights, int rows, int cols);
// void scale_scores(float *scores, int total_elements, int d_k);

void compute_mean_variance(const float *M, int size, float *mean_out,
                           float *var_out);

void apply_gelu(float *M, int rows, int cols);

#endif
