#pragma once
#ifndef STRETCHCONSTRAINT_H
#define STRETCHCONSTRAINT_H

#include "Constraint.h"

class StretchConstraint :public Constraint {
public:
	float d;
	StretchConstraint(int cardinality, float k, bool type, float d);
	float calc() override;
	float constraintFunction() override;
};

#endif