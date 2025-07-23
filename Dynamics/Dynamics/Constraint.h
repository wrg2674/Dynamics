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

	void GS_Iteration(float tstep, int ns);
	vector<glm::vec3> calcGradient(float tstep);
	void calcCentralDiff(Vertex*cur, float tstep, glm::vec3& result);
	void calcDeltaP(int idx, vector<glm::vec3>& gradient, float tstep);

	virtual void projectionFunction(float tstep, int ns);
	virtual bool satisfyConstraintFunction();
	virtual float constraintFunction()=0;


};


#endif CONSTRAINT_H