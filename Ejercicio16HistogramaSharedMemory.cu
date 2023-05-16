#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
#include <sys/time.h>
#define BLOCK_SIZE 32
#define NUM_BINS 4096
#define MAX_VAL 127

__global__ void histogramGPU(unsigned int* in, unsigned int* bins, unsigned int elems) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int i = (bx * blockDim.x) + tx;
    __shared__ int hist[NUM_BINS];

      for (int j=tx; j<NUM_BINS; j+=BLOCK_SIZE) {
            if (j < NUM_BINS) {
                hist[j] = 0;
            }
      }
    __syncthreads();

    if (i < elems) {
        atomicAdd(&(hist[in[i]]), 1);
    }
    __syncthreads();

      for (int j=tx; j<NUM_BINS; j+=BLOCK_SIZE) {
            if (j < NUM_BINS) {
                atomicAdd(&(bins[j]), hist[j]);
            }
      }
}

void histogramSecuencial(unsigned int* in, unsigned int* bins, unsigned int elems) {
    for (int i=0; i< elems; i++) {
        if (bins[in[i]] < MAX_VAL) {
            bins[in[i]]++;
        }
    }
}


int main(void) {
    struct timeval t1, t2, t3, t4;
 
    int inLength = 524288;
    unsigned int* hostBins_Secuencial;
    unsigned int* hostin; unsigned int* hostBins;
    unsigned int* devicein; unsigned int* deviceBins;


    size_t histoSize = NUM_BINS * sizeof(unsigned int);
    size_t inSize = inLength * sizeof(unsigned int);

    hostin = (unsigned int*)malloc(inSize);
    hostBins = (unsigned int*)malloc(histoSize);
    hostBins_Secuencial = (unsigned int*)malloc(histoSize);

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
    histogramGPU<<<blockPerGrid, threadPerBlock>>>(devicein, deviceBins, inLength);
    gettimeofday(&t2, 0);
    double time1 = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    cout<< "el tiempo que tardo en GPU fue"<< time1 << endl; 


    threadPerBlock.x = BLOCK_SIZE;
    blockPerGrid.x = ceil(NUM_BINS/(float)BLOCK_SIZE);

    cudaMemcpy(hostBins, deviceBins, histoSize, cudaMemcpyDeviceToHost);

    for (int i=0; i<NUM_BINS; i++) {
        hostBins_Secuencial[i] = 0;
    }
    gettimeofday(&t3, 0);
    histogramSecuencial(hostin, hostBins_Secuencial, inLength);

    gettimeofday(&t4, 0);
    double time = (1000000.0*(t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec)/1000.0;
    cout<< "el tiempo que tardo en Secuencial"<< time << endl; 


    printf("Secuencial  \n");
     for (int i=0; i<NUM_BINS; i++) {
    printf("%d, ", hostBins_Secuencial[i]);
    }
    printf("\n");
    printf("Shared memory  \n");
    for (int i=0; i<NUM_BINS; i++) {
        printf("%d, ", hostBins[i]);
    }

    free(hostBins); free(hostBins_Secuencial); free(hostin);
    cudaFree(devicein); cudaFree(deviceBins);

    return 0;
}