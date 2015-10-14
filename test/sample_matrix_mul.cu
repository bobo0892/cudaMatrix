#include "matrixMul.hpp"
#include "CUDATimer.h"

#define BLOCK_SIZE 32
#define VECTOR_SIZE 4

template <typename DataType> void rand_data(DataType *data, size_t num);

template <> void rand_data<float>(float *data, size_t num){
	for(int i = 0; i < num; ++i){
		data[i] = rand() / (float)RAND_MAX;
	}
}

template <> void rand_data<double>(double *data, size_t num){
	for(int i = 0; i < num; ++i){
		data[i] = rand() / (double)RAND_MAX;
	}
}

template <typename DataType>
void matrix_mul_cpu(const size_t M, const size_t N, const size_t K,
		const DataType *__restrict A, const DataType *__restrict B,
		DataType *__restrict C, const DataType alpha, const DataType beta) {

	for(size_t y = 0; y < M; ++y){
		for(size_t x = 0; x < N; ++x){
			
			DataType ret = 0.0;
			for(size_t z = 0; z < K; ++z){
				ret += A[y*K+z] * B[z*N+x];
			}
			C[y*N+x] = alpha * ret + beta * C[y*N+x];
		}
	}
}

template <typename DataType>
void checkResult(const size_t M, const size_t N, const DataType lamda,
		const DataType *__restrict cpu_result, const DataType *__restrict gpu_result) {

	for(size_t i = 0; i < M*N; ++i){
		if(lamda < abs(cpu_result[i]-gpu_result[i])){
			std::cout << i << " " <<  cpu_result[i] << " " << gpu_result[i] 
				<< " " << abs(cpu_result[i]-gpu_result[i]) << std::endl;
			std::cout << "gpu result is not equal to cpu result" << std::endl;
			return;
		}
	}
 	std::cout << "gpu result is equal to cpu result" << std::endl;
}
	
template<bool check>
void matrix_multiply(const size_t M, const size_t N, const size_t K, 
		const float alpha, const float beta) {

	// malloc host memory and init
	float *h_A = new float[M*K];
	rand_data<float>(h_A, M*K);
	float *h_B = new float[N*K];
	rand_data<float>(h_B, N*K);
	float *cpu_result = new float[M*N];
	// comute on cpu
	if(check)
		matrix_mul_cpu<float>(M, N, K, h_A, h_B, cpu_result, alpha, beta);

	// malloc device memory 
	float *d_A = NULL;
	size_t dataSize = sizeof(float)*M*K;
	checkCUDAError(cudaMalloc((void**)&d_A, dataSize));
	HOST2DEVICE(d_A, h_A, dataSize);
	float *d_B = NULL;
	dataSize = sizeof(float)*N*K;
	checkCUDAError(cudaMalloc((void**)&d_B, dataSize));
	HOST2DEVICE(d_B, h_B, dataSize);
	float *d_C;
	dataSize = sizeof(float)*M*N;
	checkCUDAError(cudaMalloc((void**)&d_C, dataSize));
	checkCUDAError(cudaMemset(d_C, 0, dataSize));

	int device ;
	checkCUDAError(cudaGetDevice(&device));
	cudaStream_t stream;
	checkCUDAError(cudaStreamCreate(&stream));
	// comute on gpu
	CUDATimer timer;
	timer.start();
	matrix_mul_naive(device, stream, M, N, K, d_A, d_B, d_C, alpha, beta);
	timer.stop();
	checkCUDAError(cudaGetLastError());
	float elapsedTime = timer.getElapsedSeconds();
	float gFlops = (2.0*M*N*K) / (1000*1000*1000) / elapsedTime;
	std::cout << "naive time: " << elapsedTime << "ms   "  << gFlops << "gFlops" << std::endl;
	if(check){
		float *gpu_result = new float[M*N];
		const float lamda = 0.0001;
		DEVICE2HOST(gpu_result, d_C, dataSize);
		checkResult<float>(M, N, lamda, cpu_result, gpu_result);
		delete[] gpu_result;
	}

	// shared 
	checkCUDAError(cudaMemset(d_C, 0, dataSize));
	timer.start();
	matrix_mul_shared(device, stream, M, N, K, d_A, d_B, d_C, alpha, beta);
	timer.stop();
	elapsedTime = timer.getElapsedSeconds();
	gFlops = (2.0f*M*N*K) / (1000.0*1000*1000) / elapsedTime;
	std::cout << "shared time: " << elapsedTime << "ms   "  << gFlops << "gFlops" << std::endl;
	if(check){
		float *gpu_result = new float[M*N];
		const float lamda = 0.0001;
		DEVICE2HOST(gpu_result, d_C, dataSize);
		checkResult<float>(M, N, lamda, cpu_result, gpu_result);
		delete[] gpu_result;
	}

	checkCUDAError(cudaMemset(d_C, 0, dataSize));
	timer.start();
	matrix_mul_4vector(device, stream, M, N, K, d_A, d_B, d_C, alpha, beta);
	timer.stop();
	elapsedTime = timer.getElapsedSeconds();
	gFlops = (2.0f*M*N*K) / (1000.0*1000*1000) / elapsedTime;
	std::cout << "4vector time: " << elapsedTime << "ms   "  << gFlops << "gFlops" << std::endl;
	if(check){
		float *gpu_result = new float[M*N];
		const float lamda = 0.0001;
		DEVICE2HOST(gpu_result, d_C, dataSize);
		checkResult<float>(M, N, lamda, cpu_result, gpu_result);
		delete[] gpu_result;
	}

	checkCUDAError(cudaMemset(d_C, 0, dataSize));
	timer.start();
	matrix_mul_32x8(device, stream, M, N, K, d_A, d_B, d_C, alpha, beta);
	timer.stop();
	checkCUDAError(cudaGetLastError());
	elapsedTime = timer.getElapsedSeconds();
	gFlops = (2.0f*M*N*K) / (1000.0*1000*1000) / elapsedTime;
	std::cout << "32x8 time: " << elapsedTime << "ms   "  << gFlops << "gFlops" << std::endl;
	if(check){
		float *gpu_result = new float[M*N];
		const float lamda = 0.0001;
		DEVICE2HOST(gpu_result, d_C, dataSize);
		checkResult<float>(M, N, lamda, cpu_result, gpu_result);
		delete[] gpu_result;
	}

	checkCUDAError(cudaStreamDestroy(stream));
	checkCUDAError(cudaFree(d_A));
	checkCUDAError(cudaFree(d_B));
	checkCUDAError(cudaFree(d_C));
	delete[] h_A;
	delete[] h_B;
	delete[] cpu_result;
}


int main(int argc, char **argv) {

	size_t M = 2048;
	size_t N = 2048;
	size_t K = 2048;
	float alpha = 0.5;
	float beta = 0.5;
	if(argc >= 4){
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		K = atoi(argv[3]);
	}
	if(argc >= 6){
		alpha = atof(argv[4]);
		beta = atof(argv[5]);
	}

	//matrix_multiply<true>(M, N, K, alpha, beta);
	matrix_multiply<false>(M, N, K, alpha, beta);
	return 0;
}
