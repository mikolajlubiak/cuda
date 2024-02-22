#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <stdio.h>
#include <time.h>

void add_vectors_cpu(const int *const a, const int *const b, int *const c,
                     const int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void add_vectors_gpu(const int *const a, const int *const b,
                                int *const c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int *createdata(int num) {
  int *ptr;

  ptr = (int *)malloc(sizeof(int) * num);

  if (ptr != NULL) {
    for (int i = 0; i < num; i++) {
      ptr[i] = 0;
    }
  }
  return ptr;
}

int *gendata(int num) {
  int *ptr = createdata(num);

  if (ptr != NULL) {
    for (int j = 0; j < num; j++) {
      ptr[j] = rand();
    }
  }
  return ptr;
}

int main() {
  clock_t tic;
  clock_t toc;
  const int SIZE = 1000000000; // 1_000_000_000

  int *a = gendata(SIZE);
  int *b = gendata(SIZE);
  int *c = (int *)malloc(SIZE * sizeof(int));

  int *cuda_a;
  int *cuda_b;
  int *cuda_c;

  cudaMalloc(&cuda_a, SIZE * sizeof(int));
  cudaMalloc(&cuda_b, SIZE * sizeof(int));
  cudaMalloc(&cuda_c, SIZE * sizeof(int));

  cudaMemcpy(cuda_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  tic = clock();
  add_vectors_cpu(a, b, c, SIZE);
  toc = clock();
  printf("Elapsed CPU: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  tic = clock();
  add_vectors_gpu<<<1, SIZE>>>(cuda_a, cuda_b, cuda_c);
  toc = clock();
  printf("Elapsed GPU: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  cudaMemcpy(c, cuda_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  return 0;
}
