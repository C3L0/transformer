#include "../include/utils.h"

#include <math.h>
#include <stdio.h>

#define EPSILON 1e-3

int compare(float *C1, float *C2, int len) {
  for (int i = 0; i < len; i++) {
    if (fabsf(C1[i] - C2[i]) > EPSILON) {
      printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
      return 0;
    }
  }
  return 1;
}

void print_mat(const char *name, const float *arr, int size, int max_print) {
  printf("%s (size=%d): ", name, size);
  for (int i = 0; i < size && i < max_print; i++) {
    printf("%.4f ", arr[i]);
  }
  if (size > max_print)
    printf("...");
  printf("\n");
}
