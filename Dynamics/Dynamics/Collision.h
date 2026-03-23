#ifndef COLLISION_H
#define COLLISION_H

#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct CollisionPlane {
	float3 n;
	float offset;
	float friction;
	float restitution;
};

struct CollisionSphere {
	float3 c;
	float r;
	float friction;
	float restitution;
};


#endif