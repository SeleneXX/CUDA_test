#include<stdio.h>
#include<iostream>
#include <sys/time.h> 
#include <unistd.h>
using namespace std;

// vector add cuda function
__global__ void add(float* x, float * y, float* z, int n)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

// to detect error in result
void errorResultDect(float* z, int N){
    float maxError = 0.0;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    }
    if(maxError != 0.0){
        std::cout << "Errors in this result: " << maxError << std::endl;
    }
    return;
}

int main()
{
    // to store time
    struct timeval start;
    struct timeval end;
    long cpu_runtimes[20];
    long gpu_runtimes[20];
    // loop to test vector add
    for(int j = 0; j<20; j++){
        // adding the zeros after 0010 to get 2^test_time size number
        int N = 2 << j;
        int nBytes = N * sizeof(float);
        // get host memory
        float *x, *y, *z;
        x = (float*)malloc(nBytes);
        y = (float*)malloc(nBytes);
        z = (float*)malloc(nBytes);

        // initialized data
        for (int i = 0; i < N; ++i)
        {
            x[i] = 10.0;
            y[i] = 20.0;
        }

        // get device memory
        float *d_x, *d_y, *d_z;
        cudaMalloc((void**)&d_x, nBytes);
        cudaMalloc((void**)&d_y, nBytes);
        cudaMalloc((void**)&d_z, nBytes);

        // kernel configeration
        int bs =256;
        // if the size of vector is smaller than 1024(max thread in each block)
        // use less thread can decreass data accesss time
        if(N<bs){
            bs=N;
        }
        dim3 blockSize(bs,1,1);
        dim3 gridSize(ceilf(N/(float)blockSize.x),1,1);

        //  CUP vector add
        gettimeofday(&start,NULL);
        for(int i = 0; i < N; i++){
            z[i] = x[i]+y[i];
        }
        gettimeofday(&end,NULL);
        // CPU Result examnation and store runtime
        errorResultDect(z, N);
        cpu_runtimes[j] =  end.tv_usec - start.tv_usec;
        // GPU vector add 
        gettimeofday(&start,NULL); 
        //  cpoy  data form host to device
        cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice); 
        // run gpu function
        add <<< gridSize, blockSize >>>(d_x, d_y, d_z, N);
        // Synchronize GPU
        cudaDeviceSynchronize();
        // get result
        cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
        gettimeofday(&end,NULL);
        // GPU Result examnation and store runtime
        errorResultDect(z, N);
        gpu_runtimes[j] =  end.tv_usec - start.tv_usec;

        // Free Memory
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        free(x);
        free(y);
        free(z);
    }
    
    // print CPU runtime
    std::cout << "----Vector add----"<<std::endl;
    std::cout << "vector size changes from 2 to 2^20"<< std::endl;;
    std::cout << "CPU runtimes: ";
    for(int j = 0; j<20; j++){
        std::cout <<  cpu_runtimes[j] << ", ";
    }
    std::cout << std::endl;
    // print GPU runtime
    std::cout << "GPU runtimes: ";
    for(int j = 0; j<20; j++){
        std::cout <<  gpu_runtimes[j] << ", ";
    }
    std::cout << std::endl;
   
    return 0;
}