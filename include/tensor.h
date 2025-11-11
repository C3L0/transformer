// tensor structure and operations

void matmul_blocked(const float *A, const float *B, float *C, int M, int N,
                    int K);
void transpose_matrix(const float *src, float *dst, int rows, int cols);
void matrix_add_vector_bias(float *matrix, const float *bias, int M, int N);
void mattri_low(float *A, int M);
