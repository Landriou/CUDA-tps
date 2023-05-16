#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
#include <sys/time.h>
#define LENGTH 524288
#define BLOCK_SIZE 32
#define NUM_BINS 4096
#define GRID_SIZE 16


__global__ void histogram(int* device_in, int* hist_bins) {
    int blockX =(BLOCK_SIZE*BLOCK_SIZE)*blockIdx.x;
    int blockY = (BLOCK_SIZE*BLOCK_SIZE)*(GRID_SIZE)*blockIdx.y;
    int ty =  threadIdx.y*(BLOCK_SIZE);
    int i= blockX+ blockY + ty + threadIdx.x;

	if(i<LENGTH) {
		atomicAdd(&hist_bins[device_in[i]], 1);
  }
}


int main(void) {
    struct timeval t1, t2;
    int inLength = 524288;
    int* hostin;  int* hostBins;
    int* devicein;  int* deviceBins;

    size_t histoSize = NUM_BINS * sizeof( int);
    size_t inSize = inLength * sizeof( int);

    hostin = ( int*)malloc(inSize);
    hostBins = ( int*)malloc(histoSize);

    srand(clock());
    for (int i=0; i<inLength; i++) {
        hostin[i] = int((float)rand()*(NUM_BINS-1)/float(RAND_MAX));
    }

    cudaMalloc((void**)&devicein, inSize);
    cudaMalloc((void**)&deviceBins, histoSize);
    cudaMemset(deviceBins, 0, histoSize);
    cudaMemcpy(devicein, hostin, inSize, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(BLOCK_SIZE, 1, 1);
    dim3 blockPerGrid(ceil(inLength/(float)BLOCK_SIZE), 1, 1);
    gettimeofday(&t1, 0);
    histogram<<<blockPerGrid, threadPerBlock>>>(devicein, deviceBins);
    gettimeofday(&t2, 0);
    double time1 = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    cout<< "el tiempo que tardo en GPU fue"<< time1 << endl; 

    cudaMemcpy(hostBins, deviceBins, histoSize, cudaMemcpyDeviceToHost);

    printf("histogram  \n");
    for (int i=0; i<NUM_BINS; i++) {
        printf("%d, ", hostBins[i]);
    }

    free(hostBins); free(hostin);
    cudaFree(devicein); cudaFree(deviceBins);

    return 0;
}