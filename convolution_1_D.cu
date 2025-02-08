#include <iostream>
#include <cuda.h>

#define WIDTH 10        // Input array size
#define MASK_WIDTH 3    // Kernel size

using namespace std;


__global__ void convolution_1_d_basic(float * N, float *M,float*P,
int Mask_Width,int Width){

    //mapping of threads to output elements

    int i = blockIdx.x * blockDim.x +threadIdx.x;
    if (i < Width) {  // Prevent out-of-bounds memory access
        float P_value = 0;
        int N_start_point = i - (Mask_Width / 2);

        for (int j = 0; j < Mask_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
                P_value += N[N_start_point + j] * M[j];
                // Intermediate P_value is accumulated in register to save DRAM bandwidth
            }
        }
        P[i] = P_value;
    }
}

int main() {
    // Input array and mask (kernel)
    float h_N[WIDTH] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // Example input
    float h_M[MASK_WIDTH] = {0.2, 0.5, 0.2};             // Example mask
    float h_P[WIDTH];  // Output array

    // Device pointers
    float *d_N, *d_M, *d_P;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_N, WIDTH * sizeof(float));
    cudaMalloc((void**)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, WIDTH * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize = (WIDTH + blockSize - 1) / blockSize;

    // Launch kernel
    convolution_1_d_basic<<<gridSize, blockSize>>>(d_N, d_M, d_P, MASK_WIDTH, WIDTH);

    // Copy result back to host
    cudaMemcpy(h_P, d_P, WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    cout << "Output after 1D convolution: ";
    for (int i = 0; i < WIDTH; i++) {
        cout << h_P[i] << " ";
    }
    cout << endl;

    // Free GPU memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}