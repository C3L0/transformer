#include "../include/attention.h"
#include "../include/math_utils.h"
#include "../include/tensor.h"

#include <cblas.h>
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

void compute_cross_attention(const float *X_q, const float *X_kv,
                             const AttentionParams *params, float *out,
                             int L_dec, int L_enc, int d_model, int num_heads) {

  int d_k = d_model / num_heads;

  // Output buffer for all heads
  float *all_heads = calloc(L_dec * d_model, sizeof(float));
  if (!all_heads)
    return; // Error handling

  // Loop over heads
  for (int h = 0; h < num_heads; h++) {
    // Calculate offsets into the W_qkv matrix for this head
    // The full matrix W_qkv has width (3 * d_model).
    // Head h uses a slice of width (3 * d_k).

    // However, we need to access W_Q, W_K, and W_V separately.
    // Conceptually, for head 'h', the weights are at:
    // W_qkv start ptr + (h * 3 * d_model * d_k) (based on your previous logic)

    const float *W_head_start = params->W_qkv + (h * 3 * d_model * d_k);

    // Pointers to the sub-matrices within this head's block
    // Each sub-matrix is (d_model x d_k)
    const float *W_Q_head = W_head_start;                       // 1st part
    const float *W_K_head = W_head_start + (d_model * d_k);     // 2nd part
    const float *W_V_head = W_head_start + (2 * d_model * d_k); // 3rd part

    // --- 1. Compute Q, K, V ---
    float *Q = calloc(L_dec * d_k, sizeof(float));
    float *K = calloc(L_enc * d_k, sizeof(float));
    float *V = calloc(L_enc * d_k, sizeof(float));

    // Q = X_q * W_Q
    matmul_blocked(X_q, W_Q_head, Q, L_dec, d_k, d_model);

    // K = X_kv * W_K
    matmul_blocked(X_kv, W_K_head, K, L_enc, d_k, d_model);

    // V = X_kv * W_V
    matmul_blocked(X_kv, W_V_head, V, L_enc, d_k, d_model);

    // --- 2. Scores = Q * K^T ---
    // (L_dec x d_k) * (d_k x L_enc) = (L_dec x L_enc)
    float *scores = calloc(L_dec * L_enc, sizeof(float));
    float *K_T = calloc(d_k * L_enc, sizeof(float));

    transpose_matrix(K, K_T, L_enc, d_k);
    matmul_blocked(Q, K_T, scores, L_dec, L_enc, d_k);
    free(K_T);

    // --- 3. Scale ---
    scale_scores(scores, L_dec * L_enc,
                 d_k); // Pass total size to scale function

    // --- 4. Masking ---
    // CRITICAL: Cross-Attention usually does NOT mask (encoder is fully
    // visible). So we skip apply_mask() here unless you specifically want
    // causal cross-attn (rare).

    // --- 5. Softmax ---
    // Apply softmax row-wise (over the Encoder length L_enc)
    float *weights = calloc(L_dec * L_enc, sizeof(float));
    softmax_rows(scores, weights,
                 L_dec); // NOTE: Check your softmax impl to ensure it knows row
                         // length is L_enc!
    // If your softmax_rows assumes LxL square, you might need to update it to
    // take (rows, cols). For now assuming L_dec == L_enc or softmax logic
    // handles rows correctly.

    // --- 6. Weighted Sum ---
    // Out = Weights * V
    // (L_dec x L_enc) * (L_enc x d_k) = (L_dec x d_k)
    float *head_out = all_heads + h * L_dec * d_k;
    matmul_blocked(weights, V, head_out, L_dec, d_k, L_enc);

    // Cleanup temp buffers
    free(Q);
    free(K);
    free(V);
    free(scores);
    free(weights);
  }

  // --- 7. Final Projection (W_o) ---
  // Out = ConcatHeads * W_o
  matmul_blocked(all_heads, params->W_o, out, L_dec, d_model, d_model);

  free(all_heads);
}
