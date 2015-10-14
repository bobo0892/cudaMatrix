#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define divUp(x,y) (((x)+(y)-1) / (y))

using namespace std;

#define checkCUDAError(err){ \
	cudaError_t cet = err;   \
	if(cudaSuccess != cet){  \
		cout << __FILE__ << "  " << __LINE__ << ":" << cudaGetErrorString(cet) << endl; \
		exit(0); \
	} \
}

#define HOST2DEVICE(dst, src, size)\
	checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice))

#define DEVICE2HOST(dst, src, size)\
	checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost))


#endif
