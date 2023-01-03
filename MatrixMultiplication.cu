#include<stdio.h>
#include<iostream>
#include <sys/time.h> 
#include <unistd.h>
using namespace std;

// Matrix dot product cuda function
__global__ void dot(long* x, long * y, long* z, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;   
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    // check boundry conditions
    if( r < N && c < N){
        // do the multiplication for one row and col
        long value = 0;
        for(int k = 0; k < N; k++){
        value += x[r * N + k] * y[k * N + c];
        }
        // store the result
        z[r * N + c] = value;
    }
    
}


int main()
{
    //six time will be GPU faster than CPU
    // number of time of teset, the max width and length of matrix will be 2^test_time
    int test_time = 8;
    // to store time
    struct timeval start;
    struct timeval end;
    long cpu_runtimes[test_time];
    long gpu_runtimes[test_time];
    for(int j = 0; j<test_time; j++){
        // adding the zeros after 0010 to get 2^test_time size number
        int N = 2 << j;
        // calculate size of matrix
        long nBytes = N*N*sizeof(long);
        // get host memory
        long *x, *y, *z;
        x = (long*)malloc(nBytes);
        y = (long*)malloc(nBytes);
        z = (long*)malloc(nBytes);

        // initialized data
        for (int i = 0; i < N*N; ++i)
        {
            x[i] = 1;
            y[i] = 2;
        }

        // get device memory
        long *d_x, *d_y, *d_z;
        cudaMalloc((void**)&d_x, nBytes);
        cudaMalloc((void**)&d_y, nBytes);
        cudaMalloc((void**)&d_z, nBytes);

        // kernel configeration
        dim3 blockSize(32,32,1);
        dim3 gridSize(ceilf(N/(float)blockSize.x),ceilf(N/(float)blockSize.y),1 );

        //  CUP vector add
        gettimeofday(&start,NULL);
        for(int i = 0; i < N; i++){
            for(int k=0;k<N; k++){
                for(int h=0; h<N;h++){
                   z[i*N + h] += x[i*N +k]*y[k*N +h]; 
                }
            }   
        }
        gettimeofday(&end,NULL);
        // CPU Result examnation and store runtime
        cpu_runtimes[j] =  end.tv_usec - start.tv_usec;
        // GPU vector add 
        gettimeofday(&start,NULL);
        //  cpoy  data form host to device
        cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
        // run gpu function
        dot <<< gridSize, blockSize >>>(d_x, d_y, d_z, N);
        
        // Synchronize GPU and get result
        cudaDeviceSynchronize();
        cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

        gettimeofday(&end,NULL);
        // GPU Result examnation and store runtime
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
    std::cout << "---- Matrix Multiplication ----"<<std::endl;
    std::cout << "Matrix width and legnth changes from 2^1 to 2^"<<test_time<< std::endl;;
    std::cout << "CPU runtimes: ";
    for(int j = 0; j<test_time; j++){
        std::cout <<  cpu_runtimes[j] << ", ";
    }
    std::cout << std::endl;
    // print GPU runtime
    std::cout << "GPU runtimes: ";
    for(int j = 0; j<test_time; j++){
        std::cout <<  gpu_runtimes[j] << ", ";
    }
    std::cout << std::endl;
    return 0;

}