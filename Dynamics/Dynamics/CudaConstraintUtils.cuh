#pragma once
#ifndef CUDA_CONSTRAINT_UTILS_CUH
#define CUDA_CONSTRAINT_UTILS_CUH

#include "Constraint.h"
#include "CudaUtils.cuh"

#include <cuda_runtime.h>
#include <functional>

class CudaConstraintGraph {
public:
    CudaConstraintGraph() = default;
    ~CudaConstraintGraph();

    CudaConstraintGraph(const CudaConstraintGraph&) = delete;
    CudaConstraintGraph& operator=(const CudaConstraintGraph&) = delete;

    CudaConstraintGraph(CudaConstraintGraph&& other) noexcept;
    CudaConstraintGraph& operator=(CudaConstraintGraph&& other) noexcept;

    void build(const std::function<void(cudaStream_t)>& recordOperations);
    void launch(cudaStream_t stream = 0) const;
    void release();

    [[nodiscard]] bool isInitialized() const;

private:
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
};

#endif
void uploadColorBatch(const ColorBatchHost& host,ColorBatchDevice& device);