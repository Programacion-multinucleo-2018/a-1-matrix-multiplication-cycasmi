#include "common.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

void print(int *mat, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        cout << mat[i] << " ";
    }

    return;
}

void initData(int *mat, const int size)
{
    int i;

    srand (time(0));
    for(i = 0; i < size; i++)
    {
        mat[i] = rand() % 10 + 1;
    }

    return;
}

void multMatrixOnHost(int *A, int *B, int *C, const int cols,
                     const int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C[j * rows + i] = 0;
            for (int shared_dim = 0; shared_dim < cols; shared_dim++)
            {
                //dot product
                C[j * rows + i] += A[shared_dim * rows + i] * B[j * rows + shared_dim];
            }
        }
    }

    return;
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// grid 1D block 1D
__global__ void multMatrixOnGPU1D(int *A, int *B, int *C, const int cols,
                     const int rows)
{
    unsigned int ix_cols = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix_cols < cols )
        for (int iy_rows = 0; iy_rows < rows; iy_rows++)
        {
            int idx = iy_rows * cols + ix_cols;
            C[idx] = 0;
            for (int shared_dim = 0; shared_dim < cols; shared_dim++)
            {
                //dot product
                C[idx] += A[shared_dim * rows + ix_cols] * B[iy_rows * rows + shared_dim];
            }
        }
}

int main(int argc, char **argv)
{
    // set up data size of matrix
    int nx = 0;
    int ny = 0;

    if(argc < 2)
    {
        nx = ny = 2;
    }
    else
    {
        nx = ny = stoi(argv[1]);
    }

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // initialize data at host side

    initData(h_A, nxy);
    initData(h_B, nxy);
    multMatrixOnHost(h_A, h_B, hostRef, nx, ny);


    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnHost elapsed %f ms\n", duration_ms.count());

    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 256;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();

    duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");


    return (0);
}
