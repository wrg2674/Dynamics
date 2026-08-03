#pragma once
#ifndef VERTEX_H
#define VERTEX_H

#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

enum Type : unsigned int { Stretch = 0, Bending = 1 };

struct Vertex {
	std::vector<glm::vec3> pos;
	std::vector<float> invM; // 질량의 역수
	std::vector<glm::vec3> v;
	std::vector<glm::vec3> p;
	std::vector<glm::vec3> dp;
	std::vector<int> dpCount;
	std::vector<std::vector<int>> constraintIndex;
};

struct VertexDevice {
	float3* pos;
	float* invM; // 질량의 역수
	float3* v;
	float3* p;
	float3* dp; 
	int* dpCount;
	unsigned int* constraintsArray; // 정점들이 갖는 제약을 순차적으로 모은 배열
	int* constraintOffset; // 각 정점이 갖는 제약의 시작 인덱스 

	int N;
	int constraintNum;
};
// 제약의 타입과 인덱스를 int 자료형 하나로 만들어주는 코드
constexpr unsigned int TYPE_BITS = 3;
constexpr unsigned int INDEX_BITS = 32u - TYPE_BITS;                 // 29
constexpr unsigned int TYPE_SHIFT = INDEX_BITS;                      // 29
constexpr unsigned int INDEX_MASK = (1u << INDEX_BITS) - 1u;         // 0x1fffffff
__host__ __device__ __forceinline__ unsigned int packCons(Type t, unsigned int localIdx) {
	return ((unsigned int)t << TYPE_SHIFT) | (localIdx & INDEX_MASK);
}
__host__ __device__ __forceinline__ Type unpackType(unsigned int packed) {
	return (Type)(packed >> TYPE_SHIFT);
}
__host__ __device__ __forceinline__ unsigned int unpackIndex(unsigned int packed) {
	return ((unsigned int)packed & INDEX_MASK);
}
#endif