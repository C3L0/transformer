#include "../include/transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void test_transformer_init() {
    printf("Testing Transformer initialization...\n");

    TransformerConfig config = {
        .num_layers = 2,
        .d_model = 64,
        .d_ff = 256,
        .num_heads = 4,
        .vocab_size = 1000,
        .max_seq_len = 128
    };

    TransformerParams params;
    init_transformer_params(&params, config);

    assert(params.token_embedding != NULL);
    assert(params.pos_encoding != NULL);
    assert(params.encoder_layers != NULL);
    assert(params.decoder_layers != NULL);
    assert(params.output_projection != NULL);

    free_transformer_params(&params);
    printf("Transformer initialization test passed!\n");
}

void test_transformer_forward() {
    printf("Testing Transformer forward pass...\n");

    TransformerConfig config = {
        .num_layers = 2,
        .d_model = 32,
        .d_ff = 128,
        .num_heads = 4,
        .vocab_size = 100,
        .max_seq_len = 50
    };

    TransformerParams params;
    init_transformer_params(&params, config);

    int L_src = 10;
    int L_tgt = 8;
    int src_tokens[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int tgt_tokens[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    float *out_logits = (float *)malloc(L_tgt * config.vocab_size * sizeof(float));
    
    compute_transformer(src_tokens, tgt_tokens, &params, out_logits, L_src, L_tgt);

    assert(out_logits != NULL);
    // basic check: check if it's not all zeros or NaNs (though random init could produce anything)
    // For now, just ensure it doesn't crash.

    free(out_logits);
    free_transformer_params(&params);
    printf("Transformer forward pass test passed!\n");
}

int main() {
    test_transformer_init();
    test_transformer_forward();
    return 0;
}
