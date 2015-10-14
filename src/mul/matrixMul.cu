#include "matrixMul.hpp"

#define BLOCK_SIZE  32
#define VECTOR_SIZE 4

#include "mul_kernel.hpp"

#define TYPE_T DataType *__restrict__ 
template <typename DataType> 
void matix_mul_naive(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const TYPE_T A, const TYPE_T B, TYPE_T C, 
		const DataType alpha, const DataType beta) {

	checkCUDAError(cudaSetDevice(device));
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(divUp(N, BLOCK_SIZE), divUp(M, BLOCK_SIZE));
	naive<<<grid, block, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
}

template <typename DataType> 
void matix_mul_shared(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const TYPE_T A, const TYPE_T B, TYPE_T C, 
		const DataType alpha, const DataType beta) {

	checkCUDAError(cudaSetDevice(device));
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(divUp(N, BLOCK_SIZE), divUp(M, BLOCK_SIZE));
	shared<<<grid, block, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
}

template <typename DataType> 
void matix_mul_4vector(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const TYPE_T A, const TYPE_T B, TYPE_T C, 
		const DataType alpha, const DataType beta) {

	checkCUDAError(cudaSetDevice(device));
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(divUp(N, BLOCK_SIZE), divUp(M, BLOCK_SIZE));
	4vector<<<grid, block, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
}

template <typename DataType> 
void matix_mul_32x8(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const TYPE_T A, const TYPE_T B, TYPE_T C, 
		const DataType alpha, const DataType beta) {

	checkCUDAError(cudaSetDevice(device));
	dim3 block(BLOCK_SIZE, 8);
	dim3 grid(divUp(N, BLOCK_SIZE), divUp(M, BLOCK_SIZE));
	32x8<<<grid, block, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
}

#undef TYPE_T
