#ifndef XPBDSOLVER_CUH
#define XPBDSOLVER_CUH

#include "CommonSolver.cuh"
#include "Constraint.h"
#include "Collision.h"
#include "CudaConstraintUtils.cuh"

#include <algorithm>
#include <vector>
#include <device_launch_parameters.h>

__device__ float calcScale(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
__device__ float calcConstraintWeight(ConstraintDevice cons, int consIndex, Type type, int ns);
__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__global__ void projectCollisionConstraint(VertexDevice ver, ConstraintDevice cons, int count);
__global__ void projectSelfCollisionConstraintKernel(VertexDevice ver, ConstraintDevice cons);

void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, CudaConstraintGraph& constraintIterationGraph, std::vector<float*>& vertexSet, std::vector<float*>& prevVertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, unsigned int* d_gridHashes, float* d_totalMass, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float selfThickness, float selfStiffness, int gridCapacity, float3* forces, float k_damping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int>& stretchColorOffset, std::vector<int>& bendingColorOffset, const float friction, const float restitution);
#endif