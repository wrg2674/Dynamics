#include "Utility.cuh"
#include "Vertex.h"

#include <cmath>

__device__ float get(const float3& value, int index) {
	return (&value.x)[index];
}
__device__ void set(float3& value, int index, const float item) {
	(&value.x)[index] = item;
}
__device__ int get(const int3& value, int index) {
	return (&value.x)[index];
}
__device__ void set(int3& value, int index, const int item) {
	(&value.x)[index] = item;
}

__device__ int get(const int2& value, int index) {
	return (&value.x)[index];
}
__device__ void set(int2& value, int index, const int item) {
	(&value.x)[index] = item;
}
__device__ int get(const int4& value, int index) {
	return (&value.x)[index];
}
__device__ void set(int4& value, int index, const int item) {
	(&value.x)[index] = item;
}


__device__ float length(const float3 vec) {
	return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}
__device__ float norm2(const float3 vec) {
	return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

__device__ float3 normalize(const float3 vec) {
	float3 result = make_float3(0, 0, 0);
	float norm = length(vec);
	if (norm == 0) {
		return result;
	}
	for (int i = 0; i < 3; i++) {
		set(result, i, get(vec, i) / norm);
	}
	return result;
}


__device__ float3 cross(const float3 vec1, const float3 vec2) {
	float3 result;
	result.x = vec1.y * vec2.z - vec1.z * vec2.y;
	result.y = -(vec1.x * vec2.z - vec1.z * vec2.x);
	result.z = vec1.x * vec2.y - vec1.y * vec2.x;

	return result;
}
__device__ float dot(const float3 vec1, const float3 vec2) {
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}
__device__ float clamp(const float value, const float min, const float max) {
	if (value > max) {
		return max;
	}
	if (value < min) {
		return min;
	}
	return value;
}


__device__ float3 add(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 sub(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 add(const float3& a, const float& b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ float3 sub(const float3& a, const float& b) {
	return make_float3(a.x - b, a.y - b, a.z - b);
}
__device__ float3 mul(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float3 mul(const float3& a, const float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 pickP(const VertexDevice& ver, int idx, int overrideVid, const float3& overridePos) {
	return (idx == overrideVid) ? overridePos : ver.p[idx];
}
__device__ float3 mul(const mat3& a, const float3& b) {
	return make_float3(dot(a.row0, b), dot(a.row1, b), dot(a.row2, b));
}
__device__ mat3 mul(const mat3& a, const mat3& b) {
	mat3 result;
	float3 col0 = make_float3(b.row0.x, b.row1.x, b.row2.x);
	float3 col1 = make_float3(b.row0.y, b.row1.y, b.row2.y);
	float3 col2 = make_float3(b.row0.z, b.row1.z, b.row2.z);

	result.row0 = make_float3(dot(a.row0, col0), dot(a.row0, col1), dot(a.row0, col2));
	result.row1 = make_float3(dot(a.row1, col0), dot(a.row1, col1), dot(a.row1, col2));
	result.row2 = make_float3(dot(a.row2, col0), dot(a.row2, col1), dot(a.row2, col2));

	return result;
}
__device__ mat3 mul(const mat3& a, const float b) {
	mat3 result;
	result.row0 = make_float3(a.row0.x * b, a.row0.y * b, a.row0.z * b);
	result.row1 = make_float3(a.row1.x * b, a.row1.y * b, a.row1.z * b);
	result.row2 = make_float3(a.row2.x * b, a.row2.y * b, a.row2.z * b);

	return result;
}
__device__ mat3 add(const mat3& a, const mat3& b) {
	mat3 result;
	result.row0 = add(a.row0, b.row0);
	result.row1 = add(a.row1, b.row1);
	result.row2 = add(a.row2, b.row2);

	return result;
}
__device__ mat3 transpose(const mat3& a) {
	mat3 result;
	result.row0 = make_float3(a.row0.x, a.row1.x, a.row2.x);
	result.row1 = make_float3(a.row0.y, a.row1.y, a.row2.y);
	result.row2 = make_float3(a.row0.z, a.row1.z, a.row2.z);

	return result;
}
__device__ mat3 inverse(const mat3& a) {
	mat3 result;
	float a00 = a.row0.x; float a01 = a.row0.y; float a02 = a.row0.z;
	float a10 = a.row1.x; float a11 = a.row1.y; float a12 = a.row1.z;
	float a20 = a.row2.x; float a21 = a.row2.y; float a22 = a.row2.z;

	float determinant = det(a);
	float invD = 1.0 / determinant;

	result.row0.x = (a11 * a22 - a12 * a21) * invD;
	result.row0.y = -(a01 * a22 - a02 * a21) * invD;
	result.row0.z = (a01 * a12 - a02 * a11) * invD;

	result.row1.x = -(a10 * a22 - a12 * a20) * invD;
	result.row1.y = (a00 * a22 - a02 * a20) * invD;
	result.row1.z = -(a00 * a12 - a02 * a10) * invD;

	result.row2.x = (a10 * a21 - a11 * a20) * invD;
	result.row2.y = -(a00 * a21 - a01 * a20) * invD;
	result.row2.z = (a00 * a11 - a01 * a10) * invD;

	return result;
}
__device__ float det(const mat3& a) {
	return a.row0.x * ((a.row1.y * a.row2.z) - (a.row2.y * a.row1.z))
		- a.row0.y * ((a.row1.x * a.row2.z) - (a.row2.x * a.row1.z))
		+ a.row0.z * ((a.row1.x * a.row2.y) - (a.row2.x * a.row1.y));
}
__device__ void atomicAddFloat3(float3* arr, int idx, const float3& v) {
	atomicAdd(&arr[idx].x, v.x);
	atomicAdd(&arr[idx].y, v.y);
	atomicAdd(&arr[idx].z, v.z);
}
__device__ void atomicAddFloat3(float3* arr, const float3& v) {
	atomicAdd(&arr->x, v.x);
	atomicAdd(&arr->y, v.y);
	atomicAdd(&arr->z, v.z);
}
__global__ void clearVectorKernel(float3* buf, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	buf[i] = make_float3(0.0f, 0.0f, 0.0f);
}
__device__ float calcTriangleArea(const float3 a, const float3 b, const float3 c) {
	float area = 0.5 * length(cross(sub(b, a), sub(c, a)));
	return area;
}
__device__ float3 barycentric(const float3 a, const float3 b, const float3 c, const float3 p) {
	float totalArea = calcTriangleArea(a, b, c);

	if (totalArea < 1e-12f || !isfinite(totalArea)) {
		return make_float3(1.0f, 0.0f, 0.0f);
	}

	float u = calcTriangleArea(p, b, c) / totalArea;
	float v = calcTriangleArea(p, c, a) / totalArea;
	float w = calcTriangleArea(p, a, b) / totalArea;

	return make_float3(u, v, w);
}
__global__ void clearIntKernel(int* buf, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n) {
		return;
	}

	buf[i] = 0;
}
__global__ void calcHashKernel(VertexDevice ver, unsigned int* gridHashes, int* gridIndices, float cellSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ver.N) {
		return;
	}

	gridIndices[i] = i;
	int3 cell = getCellCoords(ver.p[i], cellSize);
	gridHashes[i] = computeHash(cell);
}
__global__ void findCellStartEndKernel(int n, unsigned int* gridHashes, int* cellStart, int* cellEnd) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	unsigned int hash = gridHashes[i];
	unsigned int prevHash = (i == 0) ? 999999 : gridHashes[i - 1];

	if (hash != prevHash) {
		cellStart[hash] = i;
		if (i > 0) {
			cellEnd[prevHash] = i;
		}
	}
	if (i == n - 1) {
		cellEnd[hash] = i + 1;
	}
}
void updateSpatialHash(VertexDevice ver, float cellSize , unsigned int* d_gridHashes, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, int gridCapacity) {
	int threads = 256;
	int blocks = (ver.N + threads - 1) / threads;

	cudaMemset(d_cellStart, -1, sizeof(int) * gridCapacity);
	cudaMemset(d_cellEnd, -1, sizeof(int) * gridCapacity);

	calcHashKernel << <blocks, threads >> > (ver, d_gridHashes, d_gridIndices, cellSize);

	thrust::device_ptr<unsigned int> t_hashes(d_gridHashes);
	thrust::device_ptr<int> t_indices(d_gridIndices);
	thrust::sort_by_key(t_hashes, t_hashes + ver.N, t_indices);

	findCellStartEndKernel << <blocks, threads >> > (ver.N, d_gridHashes, d_cellStart, d_cellEnd);
}