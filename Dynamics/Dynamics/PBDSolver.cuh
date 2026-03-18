#ifndef PBDSOLVER_CUH
#define PBDSOLVER_CUH

#include "CommonSolver.cuh"

#include <device_launch_parameters.h>

__device__ void calcDeltaP(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
__device__ void projectionFunction(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep, int ns);
__device__ float calcScale(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep);
__device__ CollisionDetection CCD();
__device__ void generateCollisionConstraint();
__device__ void GSiteration(VertexDevice ver, ConstraintDevice cons, int verIndex, float tstep, int iterationCount);
__device__ void updateVertices(VertexDevice ver, int verIndex, int tstep);
__device__ void velocityUpdate();

__global__ void applyForceKernel(VertexDevice ver, float3* forces, int forceCount, float tstep);
__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp);
__global__ void applyDampingKernel(VertexDevice ver, DampingDevice damp, float k_damping);
__global__ void estimatePKernel(VertexDevice ver, float tstep);
__global__ void updateVerticesKernel(VertexDevice ver, float tstep);
__global__ void solveConstraintsKernel(VertexDevice ver, ConstraintDevice cons, float tstep, int iterationCount);

__device__ float calcConstraintWeight(ConstraintDevice cons, int consIndex, Type type, int ns);
__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount);
__global__ void solveStretchColorKernel(VertexDevice ver,ConstraintDevice cons,const int* constraintIds,int count,float tstep,int iterationCount);
__global__ void solveBendingColorKernel(VertexDevice ver,ConstraintDevice cons,const int* constraintIds,int count,float tstep,int iterationCount);
void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, float3* forces, float k_damping, float tstep, int iterationCount, int forceCount, int n, std::vector<int> stretchColorOffset, std::vector<int> bendingColorOffset);

#endif