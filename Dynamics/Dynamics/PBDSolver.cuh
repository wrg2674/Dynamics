#ifndef PBDSOLVER_CUH
#define PBDSOLVER_CUH

#include "CommonSolver.cuh"
#include "Constraint.h"
#include "Collision.h"

#include <algorithm>
#include <vector>
#include <device_launch_parameters.h>

__device__ void calcDeltaP(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
__device__ void projectionFunction(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep, int ns);
__device__ float calcScale(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
__device__ void GSiteration(VertexDevice ver, ConstraintDevice cons, int verIndex, float tstep, int iterationCount);
__device__ void updateVertices(VertexDevice ver, int verIndex, int tstep);


__global__ void applyForceKernel(VertexDevice ver, float3* forces, int forceCount, float tstep);
__global__ void initDampingVariablesKernel(DampingDevice damp, float* d_totalMass);
__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp, float* d_totalMass);
__global__ void finalizeDampingKernel(DampingDevice damp, float* d_totalMass);
__global__ void applyDampingKernel(VertexDevice ver, DampingDevice damp, float k_damping);
__global__ void estimatePKernel(VertexDevice ver, float tstep);
__global__ void updateVerticesKernel(VertexDevice ver, float tstep);
__global__ void solveConstraintsKernel(VertexDevice ver, ConstraintDevice cons, float tstep, int iterationCount);

__device__ float calcConstraintWeight(ConstraintDevice cons, int consIndex, Type type, int ns);
__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__global__ void solveStretchColorKernel(VertexDevice ver,ConstraintDevice cons,const int* constraintIds,int count,float tstep,int iterationCount);
__global__ void solveBendingColorKernel(VertexDevice ver,ConstraintDevice cons,const int* constraintIds,int count,float tstep,int iterationCount);
__global__ void projectCollisionConstraint(VertexDevice ver, ConstraintDevice cons, int count);
__global__ void velocityUpdateKernel(VertexDevice ver, ConstraintDevice cons, float friction, float restitution);

void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, std::vector<float*>& vertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, const int2* selfPairs, int* selfPairCount, float* d_totalMass, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float selfThickness, float selfStiffness, float3* forces, float k_damping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int> stretchColorOffset, std::vector<int> bendingColorOffset, const float friction, const float restitution);

#endif