#include "Vertex.h"

Vertex::Vertex(float px, float py, float pz, float vx, float vy, float vz, float m) {
	this->x[0] = px;
	this->x[1] = py;
	this->x[2] = pz;

	this->v[0] = vx;
	this->v[1] = vy;
	this->v[2] = vz;

	this->p[0] = px;
	this->p[1] = py;
	this->p[2] = pz;

	this->m = m;
}
void Vertex::updateP(glm::vec3 value) {
	for (int i = 0; i < 3; i++) {
		this->p[i] = value[i];
	}
}

