#include "../include/attention.h"
// gcc src/*.c -o app
#include "../include/utils.h"
// gcc src/*.c -DUSE_OPENBLAS -lopenblas -o app

#include <cblas.h>
#include <stdlib.h>
#include <string.h>

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
  softmax_rows(scores, weights, L);

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

void compute_multihead_attention(const float *X, const float *W_qkv,
                                 const float *W_o, float *out, int L,
                                 int d_model, int num_heads) {
  //--1-- Split embedding into heads
  int d_k = d_model / num_heads;

  //--2-- Allocate buffer for all heads' outputs
  float *all_heads = calloc(L * d_model, sizeof(float));

  // Loop over each head
  for (int h = 0; h < num_heads; h++) {
    const float *W_head = W_qkv + h * 3 * d_model * d_k;

    //--5-- Concatenate all heads' outputs
    float *head_out = all_heads + h * L * d_k;

    float *Q = calloc(L * d_k, sizeof(float));
    float *K = calloc(L * d_k, sizeof(float));
    float *V = calloc(L * d_k, sizeof(float));
    float *scores = calloc(L * L, sizeof(float));
    float *weights = calloc(L * L, sizeof(float));

    //--3-- Compute single-head attention num_heads times
    compute_attention_gemm(X, W_head, Q, K, V, scores, weights, head_out, L,
                           d_model, d_k);

    free(Q);
    free(K);
    free(V);
    free(scores);
    free(weights);
  }

// Apply the final output projection
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_model, d_model,
              1.0f, all_heads, d_model, W_o, d_model, 0.0f, out, d_model);
#else
  matmul_blocked(all_heads, W_o, out, L, d_model, d_model);

#endif
  free(all_heads);
}
