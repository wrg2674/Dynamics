#pragma once
#ifndef PBD_H
#define PBD_H
#include "Vertex.h"
#include "Constraint.h"
#include <vector>

enum CollisionDetection { TRUE, FALSE, FAIL };

class PBD {
private:
	vector<Vertex>& vertices;
	vector<glm::vec3>& forces;
	vector<Constraint*>& constraints;
	float tstep;
	float k_damping;
	float iterationCount;
public:
	PBD(vector<Vertex>& vertices, vector<glm::vec3>& forces, vector<Constraint*>& constraints, float tstep, float k_damping, float iterationCount);
	void solve();
	void velocityUpdate();
	void updateVertices();
	void applyForce();
	void sumExtForce(vector<glm::vec3>& result);
	void dampVelocities();
	void estimateP();
	void GSiteration();
	void generateCollisionConstraint();
	CollisionDetection CCD();
};
#endif