#pragma once
struct SimulationConfig {
	unsigned int screenWidth = 800;
	unsigned int screenHeight = 600;
	float mousePickRadius = 0.08f;

	float timestep = 0.01f;
	int substep = 2;
	int iterationCount = 10;

	int rows = 50;
	int cols = 50;

	float stretchK = 9.2e4f; 
	float bendingK = 1.0f;
	float dampingK = 1.0f; // PBD
	float stretchDamping = 1.0e1f;
	float bendingDamping = 0.05f;


	float selfCollisionRadius = 0.025f;
	float selfCollisionThickness = 0.006f;
	float selfCollisionK = 0.5f;

	float friction = 0.5f;
	float restitution = 0.1f;

	int gridCapacity = 200000;
	int maxSelfPairs = 50000;

	float floorBaseY = -1.5f;
	float floorAmplitude = 1.0f;
	float floorSpeed = 0.5f;
};