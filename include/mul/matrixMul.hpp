#ifndef _MATRIX_MUL_H_
#define _MATRIX_MUL_H_ 

#include "cuda_helper.hpp"

// naive matrix multipy
template <typename DataType>
void matrix_mul_naive(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) ;

// shared matrix multipy
template <typename DataType>
void matrix_mul_shared(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta); 

template <typename DataType>
void matrix_mul_4vector(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) ;

template <typename DataType>
void matrix_mul_32x8(int device, cudaStream_t stream, 
		const size_t M, const size_t N, const size_t K,
		const DataType *__restrict__ A, const DataType *__restrict__ B,
		DataType *__restrict__ C, const DataType alpha, const DataType beta) ;

#endif
