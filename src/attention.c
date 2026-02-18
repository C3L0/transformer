#include "../include/attention.h"
#include "../include/math_utils.h"
#include "../include/tensor.h"

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

void scale_scores(float *scores, int L, int d_k) {
  float scale = 1.0f / sqrtf((float)d_k);
  for (int i = 0; i < L * L; i++)
    scores[i] *= scale;
}

void apply_mask(float *scores, const float *mask, int L) {
  if (!mask)
    return;
  for (int i = 0; i < L * L; i++)
    if (mask[i] == 0.0)
      scores[i] = -INFINITY;
}

void compute_attention_gemm(const float *X, const float *W_qkv, float *Q,
                            float *K, float *V, float *scores, float *weights,
                            float *out, int L, int d_model, int d_k) {

  //--1-- Compute QKV
  // = X × W_qkv   (L x d_model) * (d_model x 3d_model)
  float *QKV = calloc(L * 3 * d_k, sizeof(float));
  matmul_blocked(X, W_qkv, QKV, L, 3 * d_k, d_model);

  //--2-- Split QKV into Q, K, V (L x d_k)
  int stride = 3 * d_k;
  for (int i = 0; i < L; i++) {
    memcpy(Q + i * d_k, QKV + i * stride + 0 * d_k, sizeof(float) * d_k);
    memcpy(K + i * d_k, QKV + i * stride + 1 * d_k, sizeof(float) * d_k);
    memcpy(V + i * d_k, QKV + i * stride + 2 * d_k, sizeof(float) * d_k);
  }

  //--3-- Compute scores
  // = Q × K^T : (L × d_k) * (d_k × L) = (L × L)
  //
  //
  //
  //
  // why no transpose in the useopenblas
  //
  //
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, d_k, 1.0f, Q, d_k,
              K, d_k, 0.0f, scores, L);
#else
  float *K_T = calloc(L * d_k, sizeof(float));
  transpose_matrix(K, K_T, L, d_k);
  matmul_blocked(Q, K_T, scores, L, L, d_k);
  free(K_T);

#endif
  //--4-- Scale scores
  scale_scores(scores, L, d_k);

  // --5-- Mask application
  float *mask = calloc(L * L, sizeof(float));
  mattri_low(mask, L);
  apply_mask(scores, mask, L);
  free(mask);

  //--6-- Compute the softmax
  softmax_rows(scores, weights, L, L);

  // --7-- Compute out
  // = weights × V  (L × L) * (L × d_k) = (L × d_k)
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_k, L, 1.0f,
              weights, L, V, d_k, 0.0f, out, d_k);
#else
  matmul_blocked(weights, V, out, L, L, d_k);
#endif

  free(QKV);
}

void compute_multihead_attention(const float *X, const AttentionParams *params,
                                 float *out, int L, int d_model,
                                 int num_heads) {

  // -- 1 -- Calculate dimensions
  int d_k = d_model / num_heads;

  // -- 2 -- Allocate buffer for all heads' outputs (will be concatenated)
  float *all_heads = calloc(L * d_model, sizeof(float));

  // Loop over each head
  for (int h = 0; h < num_heads; h++) {
    float *W_head = calloc(d_model * 3 * d_k, sizeof(float));

    // Copy the (3*d_k) columns for this head
    for (int i = 0; i < d_model; i++) {
      memcpy(W_head + i * (3 * d_k),
             params->W_qkv + i * (3 * d_model) + h * (3 * d_k),
             (3 * d_k) * sizeof(float));
    }

    float *head_out = all_heads + h * L * d_k;
    // Temporary buffers for single-head computation
    float *Q = calloc(L * d_k, sizeof(float));
    float *K = calloc(L * d_k, sizeof(float));
    float *V = calloc(L * d_k, sizeof(float));
    float *scores = calloc(L * L, sizeof(float));
    float *weights = calloc(L * L, sizeof(float));

    // -- 3 -- Compute single-head attention
    compute_attention_gemm(X, W_head, Q, K, V, scores, weights, head_out, L,
                           d_model, d_k);

    free(W_head);
    free(Q);
    free(K);
    free(V);
    free(scores);
    free(weights);
  }

  // -- 4 -- Concatenation is implicit in how 'all_heads' buffer was populated

  // -- 5 -- Apply the final output projection
  // = all_heads × W_o (L x d_model) * (d_model x d_model) = (L x d_model)
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_model, d_model,
              1.0f, all_heads, d_model, params->W_o, d_model, 0.0f, out,
              d_model);
#else
  matmul_blocked(all_heads, params->W_o, out, L, d_model, d_model);
#endif

  free(all_heads);
}

static void matmul_safe(const float *A, const float *B, float *C, int M, int N,
                        int K) {
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B,
              N, 0.0f, C, N);
#else
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
#endif
}

void compute_cross_attention(const float *X_q, const float *X_kv,
                             const AttentionParams *params, float *out,
                             int L_dec, int L_enc, int d_model, int num_heads) {

  int d_k = d_model / num_heads;
  float *all_heads = (float *)calloc(L_dec * d_model, sizeof(float));
  if (!all_heads)
    return;

  for (int h = 0; h < num_heads; h++) {
    // Weight Slicing
    const float *W_head_start = params->W_qkv + (h * 3 * d_model * d_k);
    const float *W_Q = W_head_start;
    const float *W_K = W_head_start + (d_model * d_k);
    const float *W_V = W_head_start + (2 * d_model * d_k);

    // 1. Q, K, V Projection
    float *Q = (float *)calloc(L_dec * d_k, sizeof(float));
    float *K = (float *)calloc(L_enc * d_k, sizeof(float));
    float *V = (float *)calloc(L_enc * d_k, sizeof(float));

    matmul_safe(X_q, W_Q, Q, L_dec, d_k, d_model);
    matmul_safe(X_kv, W_K, K, L_enc, d_k, d_model);
    matmul_safe(X_kv, W_V, V, L_enc, d_k, d_model);

    // 2. Scores = Q * K^T
    float *scores = (float *)calloc(L_dec * L_enc, sizeof(float));

#ifdef USE_OPENBLAS
    // BLAS Optimization: Use CblasTrans for the B matrix to avoid manual
    // transpose
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L_dec, L_enc, d_k,
                1.0f, Q, d_k, K, d_k, 0.0f, scores, L_enc);
#else
    float *K_T = (float *)calloc(d_k * L_enc, sizeof(float));
    transpose_matrix(K, K_T, L_enc, d_k);
    matmul_safe(Q, K_T, scores, L_dec, L_enc, d_k);
    free(K_T);
#endif

    // 3. Scale and Softmax
    scale_scores(scores, L_dec * L_enc, d_k);
    softmax_rows(scores, scores, L_dec, L_enc);

    // 4. Weighted Sum: Out = Weights * V
    float *head_out = all_heads + (h * L_dec * d_k);
    matmul_safe(scores, V, head_out, L_dec, d_k, L_enc);

    // Cleanup per head
    free(Q);
    free(K);
    free(V);
    free(scores);
  }

  // 5. Final Projection (W_o)
  matmul_safe(all_heads, params->W_o, out, L_dec, d_model, d_model);

  free(all_heads);
}
