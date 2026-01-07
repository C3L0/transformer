// multi-head attention functions
#ifndef ATTENTION_H
#define ATTENTION_H

typedef struct {
  float *W_qkv;
  float *W_o;
} AttentionParams;

void scale_scores(float *scores, int L, int d_k);

void apply_mask(float *scores, const float *mask, int L);

void compute_attention_gemm(const float *X, const float *W_qkv, float *Q,
                            float *K, float *V, float *scores, float *weights,
                            float *out, int L, int d_model, int d_k);

void compute_multihead_attention(const float *X, const AttentionParams *params,
                                 float *out, int L, int d_model, int num_heads);

void compute_cross_attention(const float *X_q, const float *X_kv,
                             const AttentionParams *params, float *out,
                             int L_dec, int L_enc, int d_model, int num_heads)

#endif
