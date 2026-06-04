#include "CudaConstraintUtils.cuh"
#include "CudaUtils.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

CudaConstraintGraph::~CudaConstraintGraph() {
	release();
}

CudaConstraintGraph::CudaConstraintGraph(CudaConstraintGraph&& other) noexcept {
	graph = std::exchange(other.graph, nullptr);
	graphExec = std::exchange(other.graphExec, nullptr);
}

CudaConstraintGraph& CudaConstraintGraph::operator=(CudaConstraintGraph&& other) noexcept {
	if (this == &other) return *this;

	release();

	graph = std::exchange(other.graph, nullptr);
	graphExec = std::exchange(other.graphExec, nullptr);

	return *this;
}

void CudaConstraintGraph::build(const std::function<void(cudaStream_t)>& recordOperations) {
	if (!recordOperations) throw std::invalid_argument("CudaConstraintGraph::build received an empty recordOperations function");

	cudaStream_t captureStream = nullptr;
	cudaGraph_t newGraph = nullptr;
	cudaGraphExec_t newGraphExec = nullptr;

	try {
		checkCuda(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags constraint graph capture stream failed");
		checkCuda(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture constraint graph failed");

		recordOperations(captureStream);

		checkCuda(cudaStreamEndCapture(captureStream, &newGraph), "cudaStreamEndCapture constraint graph failed");
		checkCuda(cudaStreamDestroy(captureStream), "cudaStreamDestroy constraint graph capture stream failed");
		captureStream = nullptr;

		checkCuda(cudaGraphInstantiate(&newGraphExec, newGraph, nullptr, nullptr, 0), "cudaGraphInstantiate constraint graph failed");
	}
	catch (...) {
		if (captureStream != nullptr) {
			cudaGraph_t abandonedGraph = nullptr;

			cudaStreamEndCapture(captureStream, &abandonedGraph);

			if (abandonedGraph != nullptr) cudaGraphDestroy(abandonedGraph);

			cudaStreamDestroy(captureStream);
		}

		if (newGraphExec != nullptr) cudaGraphExecDestroy(newGraphExec);
		if (newGraph != nullptr) cudaGraphDestroy(newGraph);

		throw;
	}

	release();

	graph = newGraph;
	graphExec = newGraphExec;
}

void CudaConstraintGraph::launch(cudaStream_t stream) const {
	if (graphExec == nullptr) throw std::runtime_error("CudaConstraintGraph::launch called before build");

	checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch constraint graph failed");
}

void CudaConstraintGraph::release() {
	if (graphExec != nullptr) {
		cudaGraphExecDestroy(graphExec);
		graphExec = nullptr;
	}

	if (graph != nullptr) {
		cudaGraphDestroy(graph);
		graph = nullptr;
	}
}

bool CudaConstraintGraph::isInitialized() const {
	return graphExec != nullptr;
}
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