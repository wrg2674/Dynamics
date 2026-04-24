#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <cuda_runtime.h>
#include <iostream>

#include "Vertex.h"

struct mat3 {
	float3 row0;
	float3 row1;
	float3 row2;
};

void checkCudaKernel(const char* msg);
__device__ float get(const float3& value, int index);
__device__ void set(float3& value, int index, const float item);
__device__ int get(const int3& value, int index);
__device__ void set(int3& value, int index, const int item);
__device__ int get(const int2& value, int index);
__device__ void set(int2& value, int index, const int item);
__device__ int get(const int4& value, int index);
__device__ void set(int4& value, int index, const int item);

__device__ float length(const float3 vec);
__device__ float norm2(const float3 vec);
__device__ float3 normalize(const float3 vec);
__device__ float3 cross(const float3 vec1, const float3 vec2);
__device__ float dot(const float3 vec1, const float3 vec2);
__device__ float clamp(const float value, const float min, const float max);
__device__ float3 add(const float3& a, const float3& b);
__device__ float3 sub(const float3& a, const float3& b);
__device__ float3 add(const float3& a, const float& b);
__device__ float3 sub(const float3& a, const float& b);
__device__ float3 mul(const float3& a, const float3& b);
__device__ float3 mul(const float3& a, const float b);
__device__ float3 pickP(const VertexDevice& ver, int idx, int overrideVid, const float3& overridePos);

__device__ float3 mul(const mat3& a, const float3& b);
__device__ mat3 mul(const mat3& a, const mat3& b);
__device__ mat3 mul(const mat3& a, const float b);
__device__ mat3 add(const mat3& a, const mat3& b);
__device__ mat3 transpose(const mat3& a);
__device__ mat3 inverse(const mat3& a);
__device__ float det(const mat3& a);

__device__ void atomicAddFloat3(float3* arr, int idx, const float3& v);
__device__ void atomicAddFloat3(float3* arr, const float3& v);
__global__ void clearVectorKernel(float3* buf, int n);
__device__ float calcTriangleArea(const float3 a, const float3 b, const float3 c);
__device__ float3 barycentric(const float3 a, const float3 b, const float3 c, const float3 p);
#endif