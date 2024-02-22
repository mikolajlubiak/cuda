#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <assert.h>
#include <stdio.h>
#include <time.h>

void add_vectors_cpu(const int *const a, const int *const b, int *const c,
                     const int num) {
  for (int i = 0; i < num; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void add_vectors_gpu(const int *const a, const int *const b,
                                int *const c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

void gendata_cpu(int *const arr, const int num) {
  for (int i = 0; i < num; i++) {
    arr[i] = i;
  }
}

__global__ void gendata_gpu(int *const arr) {
  int i = threadIdx.x;
  arr[i] = i;
}

int main() {
  clock_t tic;
  clock_t toc;
  const int SIZE = 100000000; // 100_000_000
  const int MEM_SIZE = SIZE * sizeof(int);
  const int BLOCK_SIZE = 256;
  const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int *a = (int *)malloc(MEM_SIZE);
  int *b = (int *)malloc(MEM_SIZE);
  int *c = (int *)malloc(MEM_SIZE);

  int *cuda_a;
  int *cuda_b;
  int *cuda_c;
  cudaMalloc(&cuda_a, MEM_SIZE);
  cudaMalloc(&cuda_b, MEM_SIZE);
  cudaMalloc(&cuda_c, MEM_SIZE);

  // CPU
  tic = clock();

  gendata_cpu(a, SIZE);
  gendata_cpu(b, SIZE);

  add_vectors_cpu(a, b, c, SIZE);

  toc = clock();

  printf("Elapsed CPU: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  int test_cpu = c[SIZE - 1];

  // GPU
  tic = clock();

  gendata_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(a);
  gendata_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(b);

  cudaMemcpy(cuda_a, a, MEM_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_b, b, MEM_SIZE, cudaMemcpyHostToDevice);

  add_vectors_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_a, cuda_b, cuda_c);

  toc = clock();

  printf("Elapsed GPU: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  cudaMemcpy(c, cuda_c, MEM_SIZE, cudaMemcpyDeviceToHost);
  int test_gpu = c[SIZE - 1];

  printf("%d, %d\n", test_cpu, test_gpu);
  assert(test_cpu == test_gpu);

  return 0;
}
