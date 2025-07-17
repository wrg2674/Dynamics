#include "distanceConstriant.h"
#include <cmath>
using namespace std;

DistanceConstraint::DistanceConstraint(float d) {
	this->d = d;
}

float DistanceConstraint::constraintFunction() {
	Vertex* a = vertices.at(0);
	Vertex* b = vertices.at(1);
	float vec[3] = { a->getP()[0] - b->getP()[0], a->getP()[1] - b->getP()[1], a->getP()[2] - b->getP()[2] };
	return getLength(vec) - d;
}

float DistanceConstraint::getLength(float vec[3]) {
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

