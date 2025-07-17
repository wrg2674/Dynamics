#pragma once

#ifndef CONSTRAINT_H
#define CONSTRAINT_H
#include <iostream>
#include <vector>
#include "Vertex.h"
using namespace std;

class Constraint {
public:
	int cardinality;
	vector<int> index;
	vector<Vertex*> vertices;
	float k;
	// true : 등식, false : 부등식
	bool type;

	Constraint();
	Constraint(int cardinality, float k, bool type);

	void addIndex(int idx);
	void addVertex(Vertex* item);

	void GS_Iteration();
	vector<float[3]> calcGradient(float tstep);
	void calcCentralDiff(Vertex*cur, float tstep, float result[]);
	void calcDeltaP();

	virtual void projectionFunction();
	virtual bool satisfyConstraintFunction();
	virtual float constraintFunction() = 0;


};


#endif CONSTRAINT_H