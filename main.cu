#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add_vectors_cpu(const int *const a, const int *const b, int *const c,
                     const int num) {
  for (int i = 0; i < num; i++) {
    c[i] = a[i] + b[i];
  }
}

void gendata_cpu(int *const arr, const int num) {
  for (int i = 0; i < num; i++) {
    arr[i] = i;
  }
}

__global__ void add_vectors_gpu(const int *const a, const int *const b,
                                int *const c, const int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num)
    c[idx] = a[idx] + b[idx];
}

__global__ void gendata_gpu(int *const arr, const int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num)
    arr[idx] = idx;
}

int main() {
  cudaError_t err;

  float time;
  cudaEvent_t start_gpu, stop_gpu;
  clock_t start_cpu, stop_cpu;

  const int SIZE = 100000000; // 100_000_000
  const int MEM_SIZE = SIZE * sizeof(int);
  int block_size, grid_size;

  int *a_cpu = (int *)malloc(MEM_SIZE);
  int *b_cpu = (int *)malloc(MEM_SIZE);
  int *c_cpu = (int *)malloc(MEM_SIZE);
  int *a_gpu;
  int *b_gpu;
  int *c_gpu;

  err = cudaMalloc(&a_gpu, MEM_SIZE);
  if (err != cudaSuccess) {
    printf("Failed to allocate memory: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }
  cudaMalloc(&b_gpu, MEM_SIZE);
  cudaMalloc(&c_gpu, MEM_SIZE);

  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  err = cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, gendata_gpu,
                                           0, SIZE);
  if (err != cudaSuccess) {
    printf("Failed to determine max potential block size: %s\n",
           cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  grid_size = (SIZE + block_size - 1) / block_size;

  // CPU
  start_cpu = clock();

  gendata_cpu(a_cpu, SIZE);
  gendata_cpu(b_cpu, SIZE);

  add_vectors_cpu(a_cpu, b_cpu, c_cpu, SIZE);

  stop_cpu = clock();

  printf("Elapsed CPU: %f seconds\n",
         (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC);

  int assert_cpu = c_cpu[SIZE - 1];

  // GPU
  cudaEventRecord(start_gpu, 0);

  gendata_gpu<<<grid_size, block_size>>>(a_gpu, SIZE);
  gendata_gpu<<<grid_size, block_size>>>(b_gpu, SIZE);

  add_vectors_gpu<<<grid_size, block_size>>>(a_gpu, b_gpu, c_gpu, SIZE);

  cudaEventRecord(stop_gpu, 0);
  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&time, start_gpu, stop_gpu);

  printf("Elapsed GPU: %f seconds\n", time / 1000);

  cudaMemcpy(c_cpu, c_gpu, MEM_SIZE, cudaMemcpyDeviceToHost);
  int assert_gpu = c_cpu[SIZE - 1];

  printf("%d, %d\n", assert_cpu, assert_gpu);
  assert(assert_cpu == assert_gpu);

  return EXIT_SUCCESS;
}
