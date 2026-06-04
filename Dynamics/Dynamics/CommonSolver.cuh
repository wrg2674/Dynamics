#ifndef COMMONSOLVER_CUH
#define COMMONSOLVER_CUH

#include "Vertex.h"
#include "Constraint.h"
#include "Utility.cuh"
#include "Damping.h"
#include "Collision.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

const int CCD_INTERVAL = 4;

__device__ void calcCentralDiff(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result);
__device__ void calcStretchGradient(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float3& result);
__device__ void calcBendingGradient(VertexDevice ver, ConstraintDevice cons, int consIndex, float3* result);
__device__ void calcGradient(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result);
__device__ float stretch_impl(VertexDevice ver, ConstraintDevice cons, int consIndex, int overrideVid, const float3& overridePos);
__device__ float calcStretch(VertexDevice ver, ConstraintDevice cons, int consIndex);
__device__ float calcStretchOverride(VertexDevice ver, ConstraintDevice cons, int consIndex, int verIndex, const float3& newPos);
__device__ float bending_impl(VertexDevice ver, ConstraintDevice cons, int consIndex, int overrideVid, const float3& overridePos);
__device__ float calcBending(VertexDevice ver, ConstraintDevice cons, int consIndex);
__device__ float calcBendingOverride(VertexDevice ver, ConstraintDevice cons, int consIndex, int verIndex, const float3& newPos);
__device__ float3 calcPoscm(VertexDevice ver);
__device__ float3 calcVcm(VertexDevice ver);
__device__ float3 calcOmega(VertexDevice ver, float3 pcm);
__global__ void clearForceKernel(float3* extForce, int n);
__global__ void windForceKernel(VertexDevice ver, int3* tris, int triCount, float3* extForce, float3 windVelocity, float airCoeff);
__device__ void createCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal, float3 colliderVelocity);
__device__ void createSelfCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal);
__global__ void detectContinuousCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, float* prevTriangle, unsigned int* d_triangleIndices, int triangleN, float tstep);
__global__ void detectStaticCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, float* prevTriangle, unsigned int* d_triangleIndices, int triangleN, float tstep);
__global__ void detectSelfCollisionKernel(VertexDevice ver, ConstraintDevice cons, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float thickness, float stiffness);
__device__ bool triangleSharesOneRingOfVertex(const int3& candidateTri, int queryVertex, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset);
__global__ void buildSelfPairsGPUKernel(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize);

//__device__ void calcDeltaP(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
//__device__ void projectionFunction(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep, int ns);
//__device__ float calcScale(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
//__device__ void GSiteration(VertexDevice ver, ConstraintDevice cons, int verIndex, float tstep, int iterationCount);

__global__ void applyForceKernel(VertexDevice ver, float3* forces, int forceCount, float tstep);
__global__ void applyAverageDeltaToPredictedKernel(VertexDevice ver);
__global__ void initDampingVariablesKernel(DampingDevice damp, float* d_totalMass);
__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp, float* d_totalMass);
__global__ void computeAngularDampingKernel(VertexDevice ver, DampingDevice damp);
__global__ void finalizeCenterOfMassKernel(DampingDevice damp, float* d_totalMass);
__global__ void finalizeOmegaKernel(DampingDevice damp);
__global__ void applyDampingKernel(VertexDevice ver, DampingDevice damp, float k_damping);
__global__ void estimatePKernel(VertexDevice ver, float tstep);
__global__ void updateVerticesKernel(VertexDevice ver, float tstep);
//__global__ void solveConstraintsKernel(VertexDevice ver, ConstraintDevice cons, float tstep, int iterationCount);

//__device__ float calcConstraintWeight(ConstraintDevice cons, int consIndex, Type type, int ns);
template <typename StretchProjector>
__global__ void solveStretchColorKernel(VertexDevice ver, ConstraintDevice cons, const int* constraintIds, int count, float tstep, int iterationCount, StretchProjector projector) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	int consIndex = constraintIds[idx];
	projector(ver, cons, consIndex, tstep, iterationCount);
}
template <typename BendingProjector>
__global__ void solveBendingColorKernel(VertexDevice ver, ConstraintDevice cons, const int* constraintIds, int count, float tstep, int iterationCount, BendingProjector projector) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	int consIndex = constraintIds[idx];
	projector(ver, cons, consIndex, tstep, iterationCount);
}

__global__ void velocityUpdateKernel(VertexDevice ver, ConstraintDevice cons, float friction, float restitution);

void launchBuildSelfPairs(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize);


#endif