#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;

    return 0;
}