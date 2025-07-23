#pragma once
#ifndef DISTANCECONSTRAINT_H
#define DISTANCECONSTRAINT_H
#include "Constraint.h"

class DistanceConstraint :public Constraint {
public:
	float d;
	DistanceConstraint(int cardinality, float k, bool type, float d);

	float constraintFunction() override;
};

#endif DISTANCECONSTRAINT_H