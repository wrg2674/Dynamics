#pragma once
#ifndef DISTANCECONSTRAINT_H
#define DISTANCECONSTRAINT_H
#include "Constraint.h"

class DistanceConstraint :public Constraint {
public:
	float d;
	DistanceConstraint(float d);

	float getLength(float vec[3]);

	float constraintFunction() override;
};

#endif DISTANCECONSTRAINT_H