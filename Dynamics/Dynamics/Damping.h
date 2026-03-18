#pragma once
#ifndef DAMPING_H
#define DAMPING_H

#include "Vertex.h"

struct Damping {
	glm::vec3* poscm;
	glm::vec3* vcm;
	glm::vec3* omega;
};

struct DampingDevice {
	float3* poscm;
	float3* vcm;
	float3* omega;
};

#endif