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

	float stretchK = 0.8f; //PBD
	float bendingK = 0.1f; //PBD
	float dampingK = 0.3f; // PBD

	float stretchCompliance = 1.0e-5f;
	float bendingCompliance = 2.0f;
	float mouseDragCompliance = 5.0e-5f;

	float stretchDamping = 1.0e2f;
	float bendingDamping = 0.005f;

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