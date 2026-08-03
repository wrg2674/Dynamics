#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

inline void checkCuda(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		std::string errorMessage = std::string(msg) + " : " + cudaGetErrorString(err);
		std::cerr << errorMessage << std::endl;
		throw std::runtime_error(errorMessage);
	}
}

inline void checkCudaKernel(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string errorMessage = std::string(msg) + " : " + cudaGetErrorString(err);
		std::cerr << errorMessage << std::endl;
		throw std::runtime_error(errorMessage);
	}
}