#include "../include/encoder.h"

void compute_encoder_layer(const float *X, const EncoderLayerParams *params,
                           float *out, int L, int d_model, int d_ff,
                           int num_heads) {
  /**
   * @brief Computes the forward pass of a single Encoder Layer.
   * * * The computation follows:
   * 1. Self-Attention: H1 = X + MultiHeadAttention(X)
   * 2. LayerNorm 1: H1_norm = LayerNorm(H1)
   * 3. FeedForward: H2 = H1_norm + FFN(H1_norm)
   * 4. LayerNorm 2: out = LayerNorm(H2)
   * * @param X The input tensor to the layer (L x d_model).
   * @param params The parameters for this layer.
   * @param out The output tensor (L x d_model).
   * @param L The sequence length.
   * @param d_model The feature dimension (D_MODEL).
   * @param d_ff The intermediate dimension of the FFN (D_FF).
   * @param num_heads The number of attention heads.
   */
}
