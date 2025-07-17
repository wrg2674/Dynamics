#pragma once
#ifndef VERTEX_H
#define VERTEX_H

class Vertex {
private:
	float p[3];
	float dp[3] = { 0 };
public:
	float x[3];
	float m; // Áú·®
	float v[3];

	Vertex(float px, float py, float pz, float vx, float vy, float vz, float m);

	void updateP(float value[3]);
	float* getP();
	float* getDp();
};

#endif VERTEX_H