#include <stdio.h>

__global__ void Hello(){
    printf("Hello GPU from the threadId: %d\n",threadIdx.x);
}
int main(){
    Hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
