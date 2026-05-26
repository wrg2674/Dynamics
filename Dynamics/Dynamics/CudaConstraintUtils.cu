#include "CudaConstraintUtils.cuh"

#include <cuda_runtime.h>

#include "CudaUtils.cuh"

void uploadColorBatch(const ColorBatchHost& host, ColorBatchDevice& device) {
	device.colorCount = (int)host.colorOffset.size() - 1;
	if (!host.constraintIds.empty()) {
		checkCuda(cudaMalloc(&device.constraintIds, sizeof(int) * host.constraintIds.size()), "cudaMalloc color.constraintIds failed");
		checkCuda(cudaMemcpy(device.constraintIds, host.constraintIds.data(), sizeof(int) * host.constraintIds.size(), cudaMemcpyHostToDevice), "cudaMemcpy color.constraintIds failed");
	}
	else {
		device.constraintIds = nullptr;
	}
	if (!host.colorOffset.empty()) {
		checkCuda(cudaMalloc(&device.colorOffset, sizeof(int) * host.colorOffset.size()), "cudaMalloc color.colorOffset failed");
		checkCuda(cudaMemcpy(device.colorOffset, host.colorOffset.data(), sizeof(int) * host.colorOffset.size(), cudaMemcpyHostToDevice), "cudaMemcpy color.colorOffset failed");
	}
	else {
		device.colorOffset = nullptr;
	}
}