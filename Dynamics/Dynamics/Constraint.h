#ifndef BENDINGCONSTRAINT_H
#define BENDINGCONSTRAINT_H

#include "Vertex.h"
#include <vector>
#include <cuda_runtime.h>

struct ColorBatchHost {
	std::vector<int> constraintIds;
	std::vector<int> colorOffset;
};
struct ColorBatchDevice {
	int* constraintIds;
	int* colorOffset;
	int colorCount;
};
struct StretchHost {
	std::vector<int2> ver;
	std::vector<float> k;
	std::vector<float> l0; //¼Ò¹®ÀÚ L
	ColorBatchHost color;
};
struct StretchDevice {
	int2* ver;
	float* k;
	float* l0;
	int n;
	ColorBatchDevice color;
};
struct BendingHost {
	std::vector<int4> ver;
	std::vector<float> k;
	std::vector<float> phi0;
	ColorBatchHost color;
};
struct BendingDevice {
	int4* ver;
	float* k;
	float* phi0;
	int n;
	ColorBatchDevice color;
};
struct CollisionHost {
	std::vector<int3> tri;
	std::vector<int> ver;
	std::vector<float> k;
	std::vector<float> thickness;
	std::vector<float3> q;
	std::vector<float3> normal;
	int* n;
};
struct CollisionDevice {
	int3* tri;
	int* ver;
	float* k;
	float* thickness;
	float3* q;
	float3* normal;
	int* n;
	int capacity;
};
struct SelfCollisionHost {
	std::vector<int3> tri;
	std::vector<int> ver;
	std::vector<float> thickness;
	std::vector<float3> q;
	std::vector<float3> normal;
	std::vector<int> n;
	int capacity;
};
struct SelfCollisionDevice {
	int3* tri;
	int* ver;
	float* thickness;
	float3* q;
	float3* normal;
	int* n;
	int capacity;
};
struct ConstraintHost {
	StretchHost stretch;
	BendingHost bending;
	CollisionHost collision;
};

struct ConstraintDevice {
	StretchDevice stretch;
	BendingDevice bending;
	CollisionDevice collision;
};
#endif

