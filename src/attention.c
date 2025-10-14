#include "attention.h"
// gcc src/*.c -o app
#include "math_utils.h"
// gcc src/*.c -DUSE_OPENBLAS -lopenblas -o app

#include <cblast.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void compute_attention_gemm(const float *X, const float *W_qkv, float *Q,
                            float *K, float *V, float *scores, float *weights,
                            float *out, int L, int d_model, int d_k) {
  // Compute QKV = X × W_qkv   (L x d_model) * (d_model x 3d_model)
  float *QKV = calloc(L * 3 * d_model, sizeof(float));
  matmul_blocked((float *)X, (float *)W_qkv, QKV,
                 L); // uses your matrix multiply

  // Split QKV into Q, K, V (L x d_k)
  for (int i = 0; i < L; i++) {
    memcpy(Q + i * d_k, QKV + i * 3 * d_k + 0 * d_k, sizeof(float) * d_k);
    memcpy(K + i * d_k, QKV + i * 3 * d_k + 1 * d_k, sizeof(float) * d_k);
    memcpy(V + i * d_k, QKV + i * 3 * d_k + 2 * d_k, sizeof(float) * d_k);
  }

  // scores = Q × K^T → (L × d_k) * (d_k × L) = (L × L)
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, d_k, 1.0f, Q, d_k,
              K, d_k, 0.0f, scores, L);
#else
  // my math implementation
  float *K_T = calloc(L * d_k, sizeof(float));
  transpose_matrix(K, K_T, L, d_k);
  matmul_blocked(Q, K_T, scores, L);
  free(K_T);

#endif

  // (Optional later): apply scaling, softmax, and masking here
  // For now, just copy scores into weights for next GEMM
  memcpy(weights, scores, sizeof(float) * L * L);

  // out = weights × V  (L × L) * (L × d_k) = (L × d_k)
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, d_k, L, 1.0f,
              weights, L, V, d_k, 0.0f, out, d_k);
#else
  matmul_blocked(weights, V, out, L);
#endif

  free(QKV);
}
