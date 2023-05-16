#include <iostream>
#include <sys/time.h>
using namespace std;
#define BLOCK_DIM 160

__global__ void transpose(float *o_matrix, float* i_matrix, int width, int height)
{
   unsigned int xi = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yi = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xi < width && yi < height)
   {
       unsigned int i_in  = xi + width * yi;
       unsigned int i_out = yi + height * xi;
       o_matrix[i_in] = i_matrix[i_out]; 
   }
}

int main() {
    cout<<  "inicie" << endl;
    struct timeval t1, t2;

    gettimeofday(&t1, 0);
    int s = 160;
    int size = s * s * sizeof(float);
    float * h_imatrix = NULL;
    h_imatrix = (float*) malloc (size);
    for (int j=0;j<(s*s);j++) {
        h_imatrix[j]=j+1.1f;
        
    }

    float* d_imatrix;
    float* d_omatrix;
    cudaMalloc( (void**) &d_imatrix, size);
    cudaMalloc( (void**) &d_omatrix, size);

    cudaMemcpy( d_imatrix, h_imatrix, size,
                                cudaMemcpyHostToDevice);

    dim3 grid(s / BLOCK_DIM, s / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    transpose<<< grid, threads >>>(d_omatrix, d_imatrix, s, s);


    float* h_omatrix = (float*) malloc(size);
    cudaMemcpy( h_omatrix, d_omatrix, size,
                                cudaMemcpyDeviceToHost);
            
  gettimeofday(&t2, 0);

      for (int j=0;j<(s*s);j++) {
       cout<<  h_omatrix[j]<< endl;
    }

double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   cout<< "el tiempo que tardo fue "<< time << endl; 

  return 0;
}