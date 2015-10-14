// naive matrix multipy
#ifndef _MUL_KERNEL_H_
#define _MUL_KERNEL_H_

template <typename DataType>
__global__ void naive(const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) {
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= N || y > M) return;

	DataType ret = 0.0;
	for(int z = 0; z < K; ++z){
		ret += A[y*K+z] * B[z*N+x];
		__syncthreads();
	}
	C[y*N+x] = alpha*ret + beta*C[y*N+x];
}

// shared matrix multipy
template <typename DataType>
__global__ void shared(const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) {
	
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = BLOCK_SIZE*K*by;
	int aEnd   = aBegin+K-1;
	int aStep  = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE*bx;
	int bStep  = BLOCK_SIZE*N;
	__shared__ DataType tempA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DataType tempB[BLOCK_SIZE][BLOCK_SIZE];

	DataType ret = 0.0;
	for(int a = aBegin, b = bBegin; a < aEnd; a+=aStep, b+=bStep){
		tempA[ty][tx] = A[a+ty*K+tx];
		tempB[ty][tx] = B[b+ty*N+tx];
		__syncthreads();
		for(int j = 0; j < BLOCK_SIZE; ++j){
			ret += tempA[ty][j] * tempB[j][tx];
			__syncthreads();
		}
	}
	DataType t = C[(by*N*BLOCK_SIZE+bx*BLOCK_SIZE)+ty*N+tx];
	C[(by*N*BLOCK_SIZE+bx*BLOCK_SIZE)+ty*N+tx] = alpha*ret + beta*t;
}

// shared matrix multipy
template <typename DataType>
__global__ void 4vector(const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) {
	
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = BLOCK_SIZE*K*by;
	int aEnd   = aBegin+K-1;
	int aStep  = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE*bx;
	int bStep  = BLOCK_SIZE*N;
	__shared__ DataType tempA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DataType tempB[BLOCK_SIZE][BLOCK_SIZE];

	DataType ret = 0.0;
	for(int a = aBegin, b = bBegin; a < aEnd; a+=aStep, b+=bStep){
		tempA[ty][tx] = A[a+ty*K+tx];
		tempB[ty][tx] = B[b+ty*N+tx];
		__syncthreads();
		#pragma unroll
		for(int j = 0; j < BLOCK_SIZE; j+=VECTOR_SIZE){
			ret += tempA[ty][j]*tempB[j][tx];
			ret += tempA[ty][j+1]*tempB[j+1][tx];
			ret += tempA[ty][j+2]*tempB[j+2][tx];
			ret += tempA[ty][j+3]*tempB[j+3][tx];
		}
		__syncthreads();
	}
	int c = N*BLOCK_SIZE*by + BLOCK_SIZE*bx;
	C[c+N*ty+tx] = alpha*ret + beta*C[c+N*ty+tx];
}

template <typename DataType>
__global__ void 32x8(const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) {
	
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = BLOCK_SIZE*K*by;
	int aEnd   = aBegin+K-1;
	int aStep  = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE*bx;
	int bStep  = BLOCK_SIZE*N;
	DataType ret[4] = {0, 0, 0, 0};
	__shared__ DataType tempA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DataType tempB[BLOCK_SIZE][BLOCK_SIZE];
	for(int a = aBegin, b = bBegin; a < aEnd; a+=aStep, b+=bStep){

		tempA[ty][tx] = A[a+ty*K+tx];
		tempB[ty][tx] = B[b+ty*N+tx];
		tempA[ty+8][tx] = A[a+(ty+8)*K+tx];
		tempB[ty+8][tx] = B[b+(ty+8)*N+tx];
		tempA[ty+16][tx] = A[a+(ty+16)*K+tx];
		tempB[ty+16][tx] = B[b+(ty+16)*N+tx];
		tempA[ty+24][tx] = A[a+(ty+24)*K+tx];
		tempB[ty+24][tx] = B[b+(ty+24)*N+tx];
		__syncthreads();
		#pragma unroll
		for(int j = 0; j < BLOCK_SIZE; j+=4){
			ret[0] += tempA[ty][j]*tempB[j][tx];
			ret[1] += tempA[ty+8][j]*tempB[j][tx];
			ret[2] += tempA[ty+16][j]*tempB[j][tx];
			ret[3] += tempA[ty+24][j]*tempB[j][tx];
			ret[0] += tempA[ty][j+1]*tempB[j+1][tx];
			ret[1] += tempA[ty+8][j+1]*tempB[j+1][tx];
			ret[2] += tempA[ty+16][j+1]*tempB[j+1][tx];
			ret[3] += tempA[ty+24][j+1]*tempB[j+1][tx];
			ret[0] += tempA[ty][j+2]*tempB[j+2][tx];
			ret[1] += tempA[ty+8][j+2]*tempB[j+2][tx];
			ret[2] += tempA[ty+16][j+2]*tempB[j+2][tx];
			ret[3] += tempA[ty+24][j+2]*tempB[j+2][tx];
			ret[0] += tempA[ty][j+3]*tempB[j+3][tx];
			ret[1] += tempA[ty+8][j+3]*tempB[j+3][tx];
			ret[2] += tempA[ty+16][j+3]*tempB[j+3][tx];
			ret[3] += tempA[ty+24][j+3]*tempB[j+3][tx];
		}
		__syncthreads();
	}
	int c = N*BLOCK_SIZE*by + BLOCK_SIZE*bx;
	C[c+N*ty+tx] = alpha*ret[0] + beta*C[c+N*ty+tx];
	C[c+N*(ty+8)+tx] = alpha*ret[1] + beta*C[c+N*(ty+8)+tx];
	C[c+N*(ty+16)+tx] = alpha*ret[2] + beta*C[c+N*(ty+16)+tx];
	C[c+N*(ty+24)+tx] = alpha*ret[3] + beta*C[c+N*(ty+24)+tx];
}

template __global__ 
void naive<float>( const size_t M, const size_t N, const size_t K,
		const float *__restrict__ A, const float *__restrict__ B,
		float *__restrict__ C, const float alpha, const float beta); 

template __global__ 
void naive<double>(const size_t M, const size_t N, const size_t K,
		const double *__restrict__ A, const double *__restrict__ B,
		double *__restrict__ C, const double alpha, const double beta); 

template __global__ 
void shared<float>(const size_t M, const size_t N, const size_t K,
		const float *__restrict__ A, const float *__restrict__ B,
		float *__restrict__ C, const float alpha, const float beta); 

template __global__ 
void shared<double>(const size_t M, const size_t N, const size_t K,
		const double *__restrict__ A, const double *__restrict__ B,
		double *__restrict__ C, const double alpha, const double beta); 

template __global__ 
void 4vector<float>(const size_t M, const size_t N, const size_t K,
		const float *__restrict__ A, const float *__restrict__ B,
		float *__restrict__ C, const float alpha, const float beta); 

template __global__ 
void 4vector<double>(const size_t M, const size_t N, const size_t K,
		const double *__restrict__ A, const double *__restrict__ B,
		double *__restrict__ C, const double alpha, const double beta); 

template __global__ 
void 32x8<float>(const size_t M, const size_t N, const size_t K,
		const float *__restrict__ A, const float *__restrict__ B,
		float *__restrict__ C, const float alpha, const float beta); 

template __global__ 
void 32x8<double>(const size_t M, const size_t N, const size_t K,
		const double *__restrict__ A, const double *__restrict__ B,
		double *__restrict__ C, const double alpha, const double beta); 

#endif
