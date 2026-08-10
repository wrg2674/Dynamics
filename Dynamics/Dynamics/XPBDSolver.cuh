#ifndef XPBDSOLVER_CUH
#define XPBDSOLVER_CUH

#include "CommonSolver.cuh"
#include "Constraint.h"
#include "Collision.h"
#include "CudaConstraintUtils.cuh"

#include <algorithm>
#include <vector>
#include <device_launch_parameters.h>

namespace xpbd {

	__global__ void sumForceKernel(const float3* forces, int forceCount, float3* totalForce);
	__global__ void predictPositionKernel(VertexDevice ver, const float3* totalForce, float tstep);

	__device__ float calcScaledCompliance(float compliance, float tstep);
	__device__ float calcDampingGamma(float compliance, float dampingStiff, float tstep);
	__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float dampingStiff, float tstep, int iterationCount, float& lambda);
	__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float dampingStiff, float tstep, int iterationCount, float& lambda);

	__global__ void projectCollisionConstraint(VertexDevice ver, ConstraintDevice cons, float tstep);
	__global__ void projectSelfCollisionConstraintKernel(VertexDevice ver, ConstraintDevice cons , float tstep);

	__global__ void projectMouseDragConstraintKernel(VertexDevice ver, int vertexIndex, float3 target, float compliance, float tstep, float3* lambda);

	void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, CudaConstraintGraph& constraintIterationGraph, std::vector<float*>& vertexSet, std::vector<float*>& prevVertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, unsigned int* d_gridHashes, float* d_totalMass, float3* d_totalForce, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float selfThickness, float selfStiffness, int gridCapacity, float3* forces, float stretchDamping, float bendingDamping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int>& stretchColorOffset, std::vector<int>& bendingColorOffset, const float friction, const float restitution, bool mouseDragActive, int mouseDragVertex, float3 mouseDragTarget, float mouseDragCompliance, float3* mouseDragLambda);

	struct StretchProjector {
		float* lambdas;
		float dampingStiff;
		__device__ __forceinline__ void operator()(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) const {
			projectStretchConstraint(ver, cons, consIndex, dampingStiff, tstep, iterationCount, lambdas[consIndex]);
		}
	};
	struct BendingProjector {
		float* lambdas;
		float dampingStiff;
		__device__ __forceinline__ void operator()(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) const {
			projectBendingConstraint(ver, cons, consIndex, dampingStiff, tstep, iterationCount, lambdas[consIndex]);
		}
	};
}

#endif