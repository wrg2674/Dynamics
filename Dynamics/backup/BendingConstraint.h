#ifndef BENDINGCONSTRAINT_H
#define BENDINGCONSTRAINT_H

#include "Constraint.h"

class BendingConstraint :public Constraint {
public:
	float phi;
	BendingConstraint(int cardinality, float k, bool type, float phi);
	float calc() override;
	float constraintFunction() override;
};

#endif

