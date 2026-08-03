#pragma once
#ifndef DAMPING_H
#define DAMPING_H

#include "Vertex.h"

constexpr int DAMPING_REDUCTION_THREADS = 256;

struct DampingCenterPartial {
	float3 weightedPosition;
	float3 weightedVelocity;
	float totalMass;
};
struct DampingAngularPartial {
	float3 angularMomentum;
	float inertia[9];
};

struct Damping {
	glm::vec3* poscm;
	glm::vec3* vcm;
	glm::vec3* omega;
	glm::vec3* angularMomentum;
	float* inertia;
};

struct DampingDevice {
	float3* poscm;
	float3* vcm;
	float3* omega;
	float3* angularMomentum;
	float* inertia;

	DampingCenterPartial* centerPartials;
	DampingAngularPartial* angularPartials;

	int partialCount;
};

#endif