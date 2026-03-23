#ifndef BENDINGCONSTRAINT_H
#define BENDINGCONSTRAINT_H

#include "Vertex.h"
#include <vector>
#include <cuda_runtime.h>

enum CollisionDetection { CollisionTrue, CollisionFalse, DetectFail };

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
	std::vector<int> hit;
	std::vector<int2> ver;
	std::vector<float3> collNormal;
	std::vector<float3> collPoint;
};
struct CollisionDevice {
	int* hit;
	int2* ver;
	float3* collNormal;
	float3* collPoint;
};
struct ConstraintHost {
	StretchHost stretch;
	BendingHost bending;
};

struct ConstraintDevice {
	StretchDevice stretch;
	BendingDevice bending;
};
#endif

