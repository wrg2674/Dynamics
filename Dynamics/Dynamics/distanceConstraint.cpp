#include "distanceConstriant.h"
#include <cmath>
using namespace std;

DistanceConstraint::DistanceConstraint(int cardinality, float k, bool type, float d):Constraint(cardinality, k, type) {
	this->d = d;
}

float DistanceConstraint::constraintFunction() {
	Vertex* a = vertices.at(0);
	Vertex* b = vertices.at(1);
	glm::vec3 vec = { a->p[0] - b->p[0], a->p[1] - b->p[1], a->p[2] - b->p[2] };
	return vec.length() - d;
}


