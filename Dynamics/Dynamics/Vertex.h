#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#include <glm/glm.hpp>

class Vertex {
private:
	
public:
	glm::vec3 x;
	float m; // Áú·®
	glm::vec3 v;
	glm::vec3 p;
	glm::vec3 dp = { 0,0,0 };
	Vertex(float px, float py, float pz, float vx, float vy, float vz, float m);

	void updateP(glm::vec3 value);

};

#endif VERTEX_H