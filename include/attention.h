// multi-head attention functions

void compute_attention_gemm(const float *X, const float *W_qkv, float *Q,
                            float *K, float *V, float *scores, float *weights,
                            float *out, int L, int d_model, int d_k);

void compute_multihead_attention(const float *X, const float *W_qkv,
                                 const float *W_o, float *out, int L,
                                 int d_model, int num_heads);
