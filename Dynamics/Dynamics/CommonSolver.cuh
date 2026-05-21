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
__device__ void createCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal);
__device__ void createSelfCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal);
__global__ void detectContinuousCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, unsigned int* d_triangleIndices, int triangleN);
__global__ void detectStaticCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, unsigned int* d_triangleIndices, int triangleN);
__global__ void detectSelfCollisionKernel(VertexDevice ver, ConstraintDevice cons, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float thickness, float stiffness);
__device__ bool triangleSharesOneRingOfVertex(const int3& candidateTri, int queryVertex, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset);
__global__ void buildSelfPairsGPUKernel(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize);
void launchBuildSelfPairs(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize);
__global__ void updateFloorKernel(float* d_floorVertices, float floorY);
void launchUpdateFloor(float* d_floorVertices, float time);

#endif